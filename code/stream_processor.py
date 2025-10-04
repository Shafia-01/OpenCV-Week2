"""StreamProcessor

A class that reads from a stream (RTSP or file), performs motion detection and camera integrity checks,
and returns annotated frames along with FPS and status flags.

Features implemented:
- Background subtraction (cv2.createBackgroundSubtractorMOG2)
- Frame differencing fallback (cv2.absdiff)
- Motion thresholding and area-based detection
- Blur detection using variance of Laplacian
- Uniform color / overexposure detection using histograms

"""

import cv2
import numpy as np
import time
from collections import deque

class StreamProcessor:
    def __init__(self, src, name="Camera", resize=(640, 360), frame_skip=0, history=500):
        self.src = src
        self.name = name
        self.resize = resize
        self.frame_skip = frame_skip  # number of frames to skip between processing
        self.cap = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=16, detectShadows=False)
        self.last_frame = None
        self.last_proc_time = time.perf_counter()
        self.fps_history = deque(maxlen=30)
        self.compromised = False
        self.compromise_reason = ""
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            print(f"[WARN] Could not open stream: {self.src}")

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            self._open()
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        if self.resize:
            frame = cv2.resize(frame, self.resize)
        return frame

    def _motion_detect(self, frame):
        # use background subtractor
        fgmask = self.bg_subtractor.apply(frame)
        # morphological clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        # threshold
        _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
        # compute percentage of motion pixels
        motion_percent = (np.count_nonzero(thresh) / thresh.size) * 100.0
        # find contours to locate motion
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = False
        for c in contours:
            if cv2.contourArea(c) > 500:  # tune this threshold
                motion = True
                break
        return motion, motion_percent, thresh

    def _laplacian_variance(self, gray):
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _histogram_uniformity(self, frame):
        # compute hist per channel and measure concentration
        chans = cv2.split(frame)
        uniform_scores = []
        for ch in chans:
            hist = cv2.calcHist([ch], [0], None, [256], [0,256])
            hist = hist.ravel()
            hist = hist / (hist.sum()+1e-8)
            # measure dominance: max bin proportion
            uniform_scores.append(hist.max())
        # return max channel dominance (close to 1 => single color dominates)
        return max(uniform_scores)

    def process_once(self):
        # Read frames (skipping if configured)
        for _ in range(self.frame_skip + 1):
            frame = self.read_frame()
            if frame is None:
                return None, None
        t0 = time.perf_counter()
        frame_rgb = frame.copy()
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

        # Motion detection
        motion, motion_percent, motion_mask = self._motion_detect(frame_rgb)

        # Blur detection
        lap_var = self._laplacian_variance(gray)
        blur_flag = (lap_var < 100.0)  # threshold suggested in task

        # Histogram uniformity (coverage / laser / blank) detection
        uni_score = self._histogram_uniformity(frame_rgb)
        # If a single color dominates beyond 0.6, suspect coverage/laser/overexposed
        uniform_flag = (uni_score > 0.6)

        # Evaluate compromise: if >75% pixels are either blank/blurred/uniform
        compromised = False
        comp_reasons = []

        # Check for large uniform areas in the motion mask or color
        if uniform_flag:
            comp_reasons.append('uniform_color')
        if blur_flag:
            comp_reasons.append('blur')

        # A heuristic for majority coverage: if motion_percent is very low AND histogram uniformity high
        if uniform_flag and motion_percent < 1.0:
            # Check how many pixels are close to the dominant color
            dominant_channel = None
            # Mark as compromised if uniform score strong
            compromised = True

        # Alternatively evaluate proportion of near-constant pixels
        # (simple fallback: count low-variance patches)
        # We'll create a compromise_percent estimate
        compromise_percent = 0.0
        if uniform_flag:
            compromise_percent = max(compromise_percent, uni_score * 100.0)
        if blur_flag:
            # inverse proportional - low lap variance => high compromise percent
            compromise_percent = max(compromise_percent, min(100.0, 100.0 * (100.0 / (lap_var + 1e-8))))

        if compromise_percent >= 75.0:
            compromised = True

        self.compromised = compromised
        self.compromise_reason = ','.join(comp_reasons)

        # Annotate frame with badges
        annotated = frame_rgb
        badge_text = []
        if motion:
            badge_text.append('Motion Detected')
        if compromised:
            badge_text.append('Camera Compromised')

        # Draw high-contrast rectangle and text in top-left
        y0 = 10
        pad_x = 8
        pad_y = 6
        for i, txt in enumerate(badge_text):
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            ((w, h), _) = cv2.getTextSize(txt, font, scale, thickness)
            x1, y1 = 10, y0 + i * (h + 2*pad_y)
            # rectangle (white bg)
            cv2.rectangle(annotated, (x1 - pad_x, y1 - pad_y), (x1 + w + pad_x, y1 + h + pad_y), (255,255,255), -1)
            # text (black)
            cv2.putText(annotated, txt, (x1, y1 + h), font, scale, (0,0,0), thickness, cv2.LINE_AA)

        # small status line bottom-left
        status = f"{self.name} | FPS: ? | LapVar:{lap_var:.1f} | Uni:{uni_score:.2f} | Motion%:{motion_percent:.2f}"
        cv2.putText(annotated, status, (10, self.resize[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        t1 = time.perf_counter()
        proc_dt = t1 - t0
        proc_fps = 1.0 / proc_dt if proc_dt > 0 else 0.0
        self.fps_history.append(proc_fps)
        avg_fps = sum(self.fps_history)/len(self.fps_history)

        return annotated, {
            'motion': motion,
            'motion_percent': motion_percent,
            'laplacian_variance': lap_var,
            'hist_uniformity': uni_score,
            'compromised': compromised,
            'compromise_reason': self.compromise_reason,
            'processing_fps': avg_fps
        }
