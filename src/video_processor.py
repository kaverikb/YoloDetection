import cv2
import os
from pathlib import Path
from typing import Dict
from tqdm import tqdm

class VideoProcessor:
    """Process video frames and save output"""
    
    def __init__(self, detector, config: Dict):
        """
        Args:
            detector: ObjectDetector instance
            config: Configuration dictionary
        """
        self.detector = detector
        self.config = config
        self.visualization_config = config['visualization']
    
    def process_video(self, input_path: str, output_path: str = None) -> str:
        """
        Process video: detect objects in each frame and save output
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video (optional)
        
        Returns:
            Path to output video
        """
        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video not found: {input_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}\n")
        
        # Set output path
        if output_path is None:
            input_name = Path(input_path).stem
            output_dir = self.config['video']['output_path']
            output_path = os.path.join(output_dir, f"detected_{input_name}.mp4")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Define codec and writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError("Cannot create video writer")
        
        print(f"Processing video: {input_path}")
        print(f"Output will be saved to: {output_path}\n")
        
        frame_count = 0
        frame_skip = self.config['video']['frame_skip']
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process every nth frame (skip frames to speed up)
                if frame_count % frame_skip == 0:
                    # Detect objects
                    detections = self.detector.detect(frame)
                    
                    # Draw detections
                    if self.visualization_config['draw_boxes']:
                        frame = self.detector.draw_detections(
                            frame,
                            detections,
                            thickness=self.visualization_config['box_thickness'],
                            font_scale=self.visualization_config['font_scale']
                        )
                    
                    # Add frame counter
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame
                out.write(frame)
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        
        print(f"\n✓ Video saved to: {output_path}")
        return output_path
    
    def get_frame_count(self, video_path: str) -> int:
        """Get total frame count of video"""
        cap = cv2.VideoCapture(video_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count