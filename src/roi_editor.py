import cv2
import json
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any

class ROIEditor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame = None
        self.current_roi = []
        self.rois = []
        self.roi_names = []
        self.drawing = False
        
        # Colors for visualization
        self.roi_color = (0, 255, 0)  # Green
        self.point_color = (0, 0, 255)  # Red
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI definition"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to current ROI
            self.current_roi.append([x, y])
            print(f"Point added: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Finish current ROI
            if len(self.current_roi) >= 3:
                roi_name = input(f"Enter name for ROI {len(self.rois) + 1}: ")
                self.rois.append(self.current_roi.copy())
                self.roi_names.append(roi_name)
                print(f"ROI '{roi_name}' saved with {len(self.current_roi)} points")
                self.current_roi = []
            else:
                print("ROI needs at least 3 points")
                
    def draw_rois(self, frame):
        """Draw current ROI and all saved ROIs"""
        display_frame = frame.copy()
        
        # Draw saved ROIs
        for i, roi in enumerate(self.rois):
            if len(roi) > 2:
                pts = np.array(roi, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, self.roi_color, 2)
                
                # Add ROI label
                centroid = np.mean(pts, axis=0, dtype=int).flatten()
                cv2.putText(display_frame, f"ROI: {self.roi_names[i]}", 
                          tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.roi_color, 2)
        
        # Draw current ROI being defined
        if len(self.current_roi) > 0:
            for point in self.current_roi:
                cv2.circle(display_frame, tuple(point), 5, self.point_color, -1)
            
            if len(self.current_roi) > 1:
                pts = np.array(self.current_roi, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], False, self.point_color, 2)
        
        return display_frame
    
    def run_editor(self):
        """Main editor loop"""
        if not self.cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return
        
        # Get first frame
        ret, self.frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
        
        cv2.namedWindow('ROI Editor')
        cv2.setMouseCallback('ROI Editor', self.mouse_callback)
        
        print("Instructions:")
        print("- Left click: Add point to ROI")
        print("- Right click: Finish current ROI")
        print("- 's': Save ROIs to JSON")
        print("- 'c': Clear all ROIs") 
        print("- 'q': Quit without saving")
        print("- ESC: Quit and save")
        
        while True:
            display_frame = self.draw_rois(self.frame)
            cv2.imshow('ROI Editor', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.rois = []
                self.roi_names = []
                self.current_roi = []
                print("All ROIs cleared")
            elif key == ord('s'):
                self.save_rois()
            elif key == 27:  # ESC
                self.save_rois()
                break
        
        cv2.destroyAllWindows()
        self.cap.release()
    
    def save_rois(self, output_path: str = "src/config/rois_cartagena.json"):
        """Save ROIs to JSON file"""
        roi_data = {
            "video_path": self.video_path,
            "rois": []
        }
        
        for i, roi in enumerate(self.rois):
            roi_data["rois"].append({
                "name": self.roi_names[i],
                "points": roi,
                "id": i
            })
        
        try:
            with open(output_path, 'w') as f:
                json.dump(roi_data, f, indent=2)
            print(f"ROIs saved to {output_path}")
        except Exception as e:
            print(f"Error saving ROIs: {e}")

def main():
    parser = argparse.ArgumentParser(description='ROI Editor for Video Analytics')
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--output', default='src/config/rois_cartagena.json', 
                       help='Output JSON file for ROIs')
    
    args = parser.parse_args()
    
    editor = ROIEditor(args.input)
    editor.run_editor()

if __name__ == "__main__":
    main()