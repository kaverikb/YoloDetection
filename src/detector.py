from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple

class ObjectDetector:
    """YOLO-based object detection"""
    
    def __init__(self, model_name: str = 'yolov8m.pt', confidence: float = 0.5, device: str = 'cpu'):
        """
        Args:
            model_name: YOLO model name
            confidence: Confidence threshold
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = YOLO(model_name)
        self.model.to(device)
        self.confidence = confidence
        self.device = device
        print(f"YOLOv8 loaded on {device}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame
        
        Args:
            frame: Input frame (numpy array)
        
        Returns:
            List of detections with boxes and labels
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf)
                class_id = int(box.cls)
                class_name = result.names[class_id]
                
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict],
                       thickness: int = 2, font_scale: float = 0.6) -> np.ndarray:
        """
        Draw bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            thickness: Box line thickness
            font_scale: Font size
        
        Returns:
            Frame with drawn boxes
        """
        annotated = frame.copy()
        
        # Color mapping
        colors = {
            'person': (0, 255, 0),      # Green
            'car': (0, 165, 255),       # Orange
            'truck': (255, 0, 0),       # Blue
            'bus': (255, 165, 0),       # Cyan
            'motorcycle': (0, 0, 255),  # Red
            'bicycle': (255, 0, 255)    # Magenta
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get color
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Draw background for text
            cv2.rectangle(annotated,
                         (x1, y1 - text_height - 10),
                         (x1 + text_width + 5, y1),
                         color, -1)
            
            # Draw text
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        
        return annotated