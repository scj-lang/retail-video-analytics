import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from shapely.geometry import Point, Polygon

def load_roi_config(config_path: str) -> List[Dict]:
    """Load ROI configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        return data.get('rois', [])
    except Exception as e:
        print(f"Error loading ROI config: {e}")
        return []

def save_roi_config(rois: List[Dict], config_path: str, video_path: str = ""):
    """Save ROI configuration to JSON file"""
    data = {
        "video_path": video_path,
        "rois": rois
    }
    
    try:
        # Ensure directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"ROI configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving ROI config: {e}")

def point_in_polygon(point: Tuple[float, float], polygon_points: List[List[float]]) -> bool:
    """Check if a point is inside a polygon using Shapely"""
    try:
        polygon = Polygon(polygon_points)
        return polygon.contains(Point(point))
    except Exception:
        return False

def draw_polygon_on_frame(frame: np.ndarray, points: List[List[int]], 
                         color: Tuple[int, int, int] = (0, 255, 0), 
                         thickness: int = 2, closed: bool = True) -> np.ndarray:
    """Draw a polygon on a frame"""
    if len(points) < 2:
        return frame
    
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    if closed and len(points) > 2:
        cv2.fillPoly(frame, [pts], color + (50,))  # Semi-transparent fill
        cv2.polylines(frame, [pts], True, color, thickness)
    else:
        cv2.polylines(frame, [pts], False, color, thickness)
    
    return frame

def get_bounding_box_center(bbox: Tuple[float, float, float, float], 
                           position: str = "bottom") -> Tuple[int, int]:
    """Get center point of bounding box
    
    Args:
        bbox: (x1, y1, x2, y2) bounding box coordinates
        position: 'center', 'bottom', 'top' - which part of bbox to use as center
    
    Returns:
        (x, y) center coordinates
    """
    x1, y1, x2, y2 = bbox
    
    center_x = int((x1 + x2) / 2)
    
    if position == "center":
        center_y = int((y1 + y2) / 2)
    elif position == "bottom":
        center_y = int(y2)
    elif position == "top":
        center_y = int(y1)
    else:
        center_y = int((y1 + y2) / 2)  # default to center
    
    return (center_x, center_y)

def calculate_polygon_area(points: List[List[float]]) -> float:
    """Calculate area of a polygon using Shapely"""
    try:
        polygon = Polygon(points)
        return polygon.area
    except Exception:
        return 0.0

def validate_video_path(video_path: str) -> bool:
    """Validate if video file exists and is readable"""
    if not Path(video_path).exists():
        print(f"Video file does not exist: {video_path}")
        return False
    
    # Try to open with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return False
    
    # Check if we can read at least one frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Cannot read frames from video: {video_path}")
        return False
    
    return True

def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video properties"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }
    
    cap.release()
    return info

def create_video_writer(output_path: str, width: int, height: int, fps: float):
    """Create OpenCV VideoWriter with proper codec"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def ensure_directories_exist(paths: List[str]):
    """Ensure that all specified directories exist"""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"Directory ensured: {path}")