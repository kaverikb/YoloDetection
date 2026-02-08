# Object Detection YOLO

A simple and efficient video object detection system using YOLOv8 and OpenCV.

## Features

- YOLOv8 object detection (person, car, truck, bus, motorcycle, bicycle)
- Real-time bounding box visualization
- Video processing with frame-by-frame detection
- Configurable confidence thresholds
- CPU and GPU support

## Installation

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic usage (default config):
```bash
python scripts/detect_video.py
```

This will process all videos in `data/input_videos/` and save results to `data/output_videos/`

### Process specific video:
```bash
python scripts/detect_video.py --input data/input_videos/video.mp4
```

### Process directory:
```bash
python scripts/detect_video.py --input data/input_videos
```

### Custom output path:
```bash
python scripts/detect_video.py --input video.mp4 --output results/detected.mp4
```

### Custom confidence threshold:
```bash
python scripts/detect_video.py --confidence 0.6
```

## Configuration

Edit `configs/config.yaml` to customize:
- YOLO model (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
- Confidence threshold
- Classes to detect
- Box thickness and colors
- Device (cpu or cuda)

## Project Structure
```
ObjectDetectionYOLO/
├── data/
│   ├── input_videos/    (put your videos here)
│   └── output_videos/   (output videos saved here)
├── src/
│   ├── detector.py      (YOLO detection logic)
│   ├── video_processor.py (video processing)
│   └── utils.py         (utilities)
├── scripts/
│   └── detect_video.py  (main script)
├── configs/
│   └── config.yaml      (configuration)
└── requirements.txt
```

## Output

- Annotated video with bounding boxes
- Each detection shows: class name + confidence score
- Frame counter in top-left corner
- Color-coded boxes (green=person, orange=car, etc.)

## Performance

- YOLOv8m model: ~30-50 FPS on GPU, ~5-10 FPS on CPU
- Adjust `frame_skip` in config for faster processing

