"""
OMNISCAN FORENSICS — Farneback Optical Flow & Temporal Continuity Analyzer
Reference: OmniScan Architecture & Synopsis (Video Forensics Route)

Evaluates dynamic video streams across consecutive frames to detect:
1. Inter-frame motion discontinuities (abrupt jumps or face-swap cut lines)
2. Dropped or duplicate frames
3. Unnatural static pauses or artificial temporal speed modifications
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger("omniscan.forensics.optical_flow")


class OpticalFlowAnalyzer:
    """
    Computes Farneback Dense Optical Flow between consecutive frames
    to extract motion vectors, magnitude statistics, and temporal coherence.
    """

    def __init__(self, pyr_scale: float = 0.5, levels: int = 3, winsize: int = 15,
                 iterations: int = 3, poly_n: int = 5, poly_sigma: float = 1.2):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma

    def compute_flow_between_frames(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Compute optical flow field and motion metrics between two frames.
        
        Returns:
            flow: np.ndarray (H, W, 2)
            metrics: dict
        """
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame

        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=0
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(magnitude))
        std_mag = float(np.std(magnitude))
        max_mag = float(np.max(magnitude))

        # Check for static freezing (near zero motion between frames)
        is_frozen = mean_mag < 0.05

        return flow, {
            "mean_magnitude": round(mean_mag, 4),
            "std_magnitude": round(std_mag, 4),
            "max_magnitude": round(max_mag, 4),
            "is_frozen": is_frozen,
        }

    def visualize_flow(self, flow: np.ndarray) -> np.ndarray:
        """Convert flow field to HSV color representation for visual inspection."""
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction
        hsv[..., 1] = 255                    # Saturation = max
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb_flow

    def analyze_sequence(self, frames: np.ndarray) -> dict:
        """
        Analyze temporal motion continuity across a sequence of frames.
        frames: np.ndarray of shape (N, H, W, C)
        """
        if len(frames) < 2:
            return {
                "motion_continuity_score": 1.0,
                "temporal_anomaly_detected": False,
                "dropped_or_frozen_frames": 0,
                "motion_jerk_index": 0.0,
                "mean_flow_magnitude": 0.0,
                "flow_visual_rgb": None,
            }

        magnitudes = []
        frozen_frames = 0
        last_flow = None

        for i in range(len(frames) - 1):
            flow, metrics = self.compute_flow_between_frames(frames[i], frames[i+1])
            magnitudes.append(metrics["mean_magnitude"])
            if metrics["is_frozen"]:
                frozen_frames += 1
            last_flow = flow

        # Analyze motion jerk (acceleration variations)
        # Deepfake cut-and-paste and frame dropping produce abrupt magnitude variance spikes
        mag_arr = np.array(magnitudes)
        motion_diffs = np.abs(np.diff(mag_arr)) if len(mag_arr) > 1 else np.array([0.0])
        jerk_index = float(np.mean(motion_diffs)) if len(motion_diffs) > 0 else 0.0
        jerk_std = float(np.std(motion_diffs)) if len(motion_diffs) > 0 else 0.0

        # Anomaly scoring
        # Abrupt unnatural spikes or multiple freeze pauses reduce continuity score
        anomaly_factor = min(1.0, (jerk_index * 1.5 + (frozen_frames / max(len(frames)-1, 1)) * 2.0))
        continuity_score = float(np.clip(1.0 - anomaly_factor, 0.0, 1.0))
        temporal_anomaly = continuity_score < 0.65 or frozen_frames > 2

        # Visual representation of the latest flow field
        flow_visual = self.visualize_flow(last_flow) if last_flow is not None else None

        return {
            "motion_continuity_score": round(continuity_score, 4),
            "temporal_anomaly_detected": temporal_anomaly,
            "dropped_or_frozen_frames": frozen_frames,
            "motion_jerk_index": round(jerk_index, 4),
            "mean_flow_magnitude": round(float(np.mean(mag_arr)), 4),
            "flow_visual_rgb": flow_visual,
        }
