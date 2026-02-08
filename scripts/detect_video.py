import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, get_video_files, print_summary
from src.detector import ObjectDetector
from src.video_processor import VideoProcessor


def main():
    """Main video detection script"""
    
    parser = argparse.ArgumentParser(description='YOLO Video Object Detection')
    parser.add_argument('--input', type=str, help='Input video file or directory')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--confidence', type=float, help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config('configs/config.yaml')
    print("Configuration loaded\n")
    
    # Get device
    device = config['device']
    print(f"Device: {device}\n")
    
    # Initialize detector
    print("Loading YOLO detector...")
    model_name = config['detection']['model']
    confidence = args.confidence or config['detection']['confidence']
    
    detector = ObjectDetector(
        model_name=model_name,
        confidence=confidence,
        device=device
    )
    print()
    
    # Initialize processor
    processor = VideoProcessor(detector, config)
    
    # Determine input videos
    if args.input:
        input_path = args.input
        if Path(input_path).is_file():
            video_files = [input_path]
        else:
            video_files = get_video_files(input_path)
    else:
        # Default to input_videos folder
        input_dir = config['video']['input_path']
        video_files = get_video_files(input_dir)
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video(s)\n")
    print("="*60)
    print("OBJECT DETECTION")
    print("="*60)
    
    # Process each video
    for video_file in video_files:
        try:
            output_path = args.output or None
            processor.process_video(video_file, output_path)
            print_summary(video_file, output_path or f"data/output_videos/detected_{Path(video_file).stem}.mp4", 0)
        except Exception as e:
            print(f"Error processing {video_file}: {e}\n")
    
    print("="*60)
    print("Detection completed!")
    print("="*60)


if __name__ == '__main__':
    main()