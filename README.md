# OpenCV Week 2 - Real-Time Streaming and Processing

This repository implements real-time motion detection and camera integrity checks for four RTSP streams and displays them in a 2x2 window.

## Features
- Motion detection per-stream using a background subtractor (MOG2) and frame differencing fallback.
- Camera integrity checks:
  - Blur detection using Laplacian variance.
  - Coverage / laser / uniform color detection via color-histogram uniformity.
- Visual overlays: high-contrast badge in the top-left with `Motion Detected` or `Camera Compromised`.
- FPS profiling (processing FPS vs stream FPS).

## Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- psutil (optional, for memory stats)

Install with:

```bash
pip install -r code/requirements.txt
````

## How to run

1. Edit `code/main.py` and replace the `RTSP_STREAMS` list with your four RTSP URLs (or local video files for testing).
2. Run:

```bash
python code/main.py
```

Controls:

* `q` or `ESC` - quit
* `s` - save a screenshot of the combined window into `diagrams/` (for deliverable)

## Deliverables

* `report/Report.pdf` - the comprehensive report with sources and notes.
* `diagrams/flow_diagram.png` - flow diagram and generated screenshots.

## Notes on performance

* If processing FPS falls behind streaming FPS, reduce `PROCESSING_RESOLUTION` in `main.py` or increase `frame_skip` in the stream processor.
* Use hardware acceleration (GStreamer, FFMPEG builds of OpenCV) for production-grade stream handling.