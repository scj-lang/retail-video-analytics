import cv2
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import torch
from yolox.exp import get_exp
from yolox.utils import postprocess
import supervision as sv
from shapely.geometry import Point, Polygon

class RetailTracker:
    def __init__(self, model_path: str = "yolox_nano.pth", exp_file: str = "yolox_nano", roi_config: str = None):
        """Initialize the retail tracking pipeline"""
        
        # Load YOLOX model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp = get_exp(exp_file, None)
        self.model = self.exp.get_model()
        
        # Load weights
        if model_path and Path(model_path).exists():
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
        
        self.model.to(self.device)
        self.model.eval()
        
        # Model parameters
        self.test_size = self.exp.test_size
        self.num_classes = self.exp.num_classes
        self.confthre = 0.25
        self.nmsthre = 0.45
        
        # Initialize ByteTrack
        self.tracker = sv.ByteTrack()
        
        # Load ROI configuration
        self.rois = []
        if roi_config and Path(roi_config).exists():
            self.load_rois(roi_config)
        
        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        
        # Event logging
        self.events = []
        self.person_roi_status = {}  # track which persons are in which ROIs
        
    def load_rois(self, config_path: str):
        """Load ROI configuration from JSON"""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                
            self.rois = []
            for roi_data in data.get('rois', []):
                roi_polygon = Polygon(roi_data['points'])
                self.rois.append({
                    'name': roi_data['name'],
                    'id': roi_data['id'],
                    'polygon': roi_polygon,
                    'points': roi_data['points']
                })
            
            print(f"Loaded {len(self.rois)} ROIs from {config_path}")
            
        except Exception as e:
            print(f"Error loading ROIs: {e}")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for YOLOX inference"""
        img = cv2.resize(frame, self.test_size)
        img = img.astype(np.float32)
        img /= 255.0
        img = img[:, :, ::-1]  # BGR to RGB
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).to(self.device)
        return img
    
    def detect_persons(self, frame):
        """Run YOLOX inference and filter for person class"""
        # Preprocess
        img = self.preprocess_frame(frame)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )[0]
        
        if outputs is None:
            return sv.Detections.empty()
        
        # Filter for person class (class 0 in COCO)
        person_mask = outputs[:, 6] == 0  # class 0 = person
        person_detections = outputs[person_mask]
        
        if len(person_detections) == 0:
            return sv.Detections.empty()
        
        # Scale back to original frame size
        scale_x = frame.shape[1] / self.test_size[0]
        scale_y = frame.shape[0] / self.test_size[1]
        
        person_detections[:, 0] *= scale_x  # x1
        person_detections[:, 1] *= scale_y  # y1
        person_detections[:, 2] *= scale_x  # x2
        person_detections[:, 3] *= scale_y  # y2
        
        # Convert to supervision format
        xyxy = person_detections[:, :4]
        confidence = person_detections[:, 4] * person_detections[:, 5]  # obj_conf * cls_conf
        class_id = person_detections[:, 6].astype(int)
        
        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
    
    def point_in_roi(self, point: tuple, roi_id: int) -> bool:
        """Check if a point is inside a specific ROI"""
        if roi_id < len(self.rois):
            return self.rois[roi_id]['polygon'].contains(Point(point))
        return False
    
    def detect_roi_events(self, detections, frame_number: int, timestamp: float):
        """Detect and log ROI entry/exit events"""
        
        for detection in detections:
            person_id = detection.tracker_id
            if person_id is None:
                continue
                
            # Get person center point (bottom of bounding box for floor position)
            x1, y1, x2, y2 = detection.xyxy[0]
            person_center = (int((x1 + x2) / 2), int(y2))  # bottom center
            
            # Check all ROIs
            current_rois = set()
            for roi_idx, roi in enumerate(self.rois):
                if roi['polygon'].contains(Point(person_center)):
                    current_rois.add(roi_idx)
            
            # Compare with previous status
            previous_rois = self.person_roi_status.get(person_id, set())
            
            # Detect entries
            for roi_id in current_rois - previous_rois:
                event = {
                    'timestamp': timestamp,
                    'frame_number': frame_number,
                    'person_id': person_id,
                    'event_type': 'roi_entry',
                    'roi_id': roi_id,
                    'roi_name': self.rois[roi_id]['name'],
                    'position_x': person_center[0],
                    'position_y': person_center[1]
                }
                self.events.append(event)
                print(f"Person {person_id} entered {self.rois[roi_id]['name']}")
            
            # Detect exits
            for roi_id in previous_rois - current_rois:
                event = {
                    'timestamp': timestamp,
                    'frame_number': frame_number,
                    'person_id': person_id,
                    'event_type': 'roi_exit',
                    'roi_id': roi_id,
                    'roi_name': self.rois[roi_id]['name'],
                    'position_x': person_center[0],
                    'position_y': person_center[1]
                }
                self.events.append(event)
                print(f"Person {person_id} exited {self.rois[roi_id]['name']}")
            
            # Update status
            self.person_roi_status[person_id] = current_rois
    
    def draw_rois(self, frame):
        """Draw ROIs on frame"""
        for roi in self.rois:
            points = np.array(roi['points'], np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(frame, [points], True, (0, 255, 0), 2)
            
            # Add ROI label
            centroid = np.mean(points, axis=0, dtype=int).flatten()
            cv2.putText(frame, roi['name'], tuple(centroid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def process_video(self, input_path: str, output_dir: str):
        """Main video processing pipeline"""
        
        # Setup paths
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(input_path).stem
        output_video_path = output_dir / f"{video_name}_processed.mp4"
        output_events_path = output_dir.parent / "logs" / f"{video_name}_events.csv"
        output_events_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        print(f"Processing {total_frames} frames at {fps} FPS...")
        print(f"Output video: {output_video_path}")
        print(f"Output events: {output_events_path}")
        
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_number / fps
            
            # YOLOX detection (filter for person class = 0)
            detections = self.detect_persons(frame)
            
            # Update tracker
            detections = self.tracker.update_with_detections(detections)
            
            # Process ROI events
            self.detect_roi_events(detections, frame_number, timestamp)
            
            # Annotate frame
            annotated_frame = self.box_annotator.annotate(frame.copy(), detections)
            annotated_frame = self.label_annotator.annotate(
                annotated_frame, detections, 
                labels=[f"Person {id}" if id is not None else "Person" 
                       for id in detections.tracker_id]
            )
            annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
            
            # Draw ROIs
            annotated_frame = self.draw_rois(annotated_frame)
            
            # Add frame info
            cv2.putText(annotated_frame, f"Frame: {frame_number}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Persons: {len(detections)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(annotated_frame)
            
            frame_number += 1
            
            # Progress update
            if frame_number % 100 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_number}/{total_frames})")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Save events to CSV
        if self.events:
            df = pd.DataFrame(self.events)
            df.to_csv(output_events_path, index=False)
            print(f"Saved {len(self.events)} events to {output_events_path}")
        else:
            print("No events detected")
        
        print("Processing complete!")

def main():
    parser = argparse.ArgumentParser(description='Retail Video Analytics Pipeline')
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--output', default='data/processed/', 
                       help='Output directory for processed video')
    parser.add_argument('--model', default='yolox_nano.pth', 
                       help='YOLOX model path')
    parser.add_argument('--exp', default='yolox_nano',
                       help='YOLOX experiment name')
    parser.add_argument('--rois', default='src/config/rois_cartagena.json',
                       help='ROI configuration file')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = RetailTracker(
        model_path=args.model,
        exp_file=args.exp,
        roi_config=args.rois if Path(args.rois).exists() else None
    )
    
    # Process video
    tracker.process_video(args.input, args.output)

if __name__ == "__main__":
    main()