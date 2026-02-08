import yaml
from pathlib import Path
from typing import Dict

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_video_files(directory: str) -> list:
    """
    Get all video files from directory
    
    Args:
        directory: Path to directory
    
    Returns:
        List of video file paths
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(directory).glob(f'*{ext}'))
        video_files.extend(Path(directory).glob(f'*{ext.upper()}'))
    
    return sorted([str(f) for f in video_files])

def print_summary(input_video: str, output_video: str, detection_count: int):
    """
    Print processing summary
    
    Args:
        input_video: Input video path
        output_video: Output video path
        detection_count: Total detections found
    """
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Input:  {input_video}")
    print(f"Output: {output_video}")
    print("="*60 + "\n")
    