"""Main runner: launches 4 StreamProcessor instances and displays them in a 2x2 window.

Replace RTSP_STREAMS with your actual streams (or local video files for testing).
"""

import cv2
import numpy as np
import time
import os
from stream_processor import StreamProcessor

# --- Configuration ---
RTSP_STREAMS = [
    "C:\\Users\\Shafia\\PROJECTS\\OpenCV-Week2\\code\\Video (1).mp4",
    "C:\\Users\\Shafia\\PROJECTS\\OpenCV-Week2\\code\\Video (2).mp4",
    "C:\\Users\\Shafia\\PROJECTS\\OpenCV-Week2\\code\\Video (3).mp4",
    "C:\\Users\\Shafia\\PROJECTS\\OpenCV-Week2\\code\\Video (4).mp4"
]
PROCESSING_RESOLUTION = (640, 360)
FRAME_SKIP = 0  # skip frames between processing to save CPU

SAVE_SCREENSHOT_DIR = os.path.join(os.getcwd(), 'diagrams')
if not os.path.exists(SAVE_SCREENSHOT_DIR):
    os.makedirs(SAVE_SCREENSHOT_DIR, exist_ok=True)

# create processors
processors = []
for i, src in enumerate(RTSP_STREAMS):
    name = f"CAM_{i+1}"
    sp = StreamProcessor(src, name=name, resize=PROCESSING_RESOLUTION, frame_skip=FRAME_SKIP)
    processors.append(sp)

WINDOW_NAME = 'MultiCam 2x2'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

try:
    while True:
        start = time.perf_counter()
        frames = []
        metas = []
        for sp in processors:
            annotated, meta = sp.process_once()
            if annotated is None:
                # create a placeholder empty frame
                annotated = np.zeros((PROCESSING_RESOLUTION[1], PROCESSING_RESOLUTION[0], 3), dtype=np.uint8)
                cv2.putText(annotated, 'No Frame', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                meta = {'processing_fps': 0.0}
            frames.append(annotated)
            metas.append(meta)

        # build 2x2 mosaic
        h, w = PROCESSING_RESOLUTION[1], PROCESSING_RESOLUTION[0]
        top = np.hstack((frames[0], frames[1])) if len(frames) >= 2 else np.hstack((frames[0], np.zeros_like(frames[0])))
        bot = np.hstack((frames[2], frames[3])) if len(frames) >= 4 else np.hstack((np.zeros_like(frames[0]), np.zeros_like(frames[0])))
        mosaic = np.vstack((top, bot))

        # show overall FPS (processing)
        end = time.perf_counter()
        loop_fps = 1.0 / (end - start) if (end - start) > 0 else 0.0
        cv2.putText(mosaic, f'Combined FPS: {loop_fps:.2f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow(WINDOW_NAME, mosaic)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('s'):
            # save screenshot
            fname = os.path.join(SAVE_SCREENSHOT_DIR, f'mosaic_{int(time.time())}.png')
            cv2.imwrite(fname, mosaic)
            print(f"Saved screenshot: {fname}")

finally:
    for sp in processors:
        sp.release()
    cv2.destroyAllWindows()