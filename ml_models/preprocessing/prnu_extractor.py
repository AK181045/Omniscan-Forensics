"""
OMNISCAN FORENSICS — PRNU Noise Pattern Extractor
Reference: J. Lukas, J. Fridrich, & M. Goljan (2006) - 
'Digital Camera Identification from Sensor Pattern Noise'

Extracts Photo-Response Non-Uniformity (PRNU) noise residuals to:
1. Verify physical hardware sensor fingerprints vs. synthetic AI media (which lacks physical PRNU).
2. Detect localized document / signature tampering via noise variance discontinuity across blocks.
"""

import cv2
import numpy as np
import scipy.signal
import logging

logger = logging.getLogger("omniscan.forensics.prnu")


class PRNUExtractor:
    """
    Extracts high-frequency sensor noise residuals using spatial denoising filters
    (Wiener / Wavelet residual estimation).
    """

    def __init__(self, block_size: int = 32):
        self.block_size = block_size

    def extract_noise_residual(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the high-frequency sensor noise residual W = I - F(I),
        where F(I) is a denoised estimate of the original image.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray_f = gray.astype(np.float32)

        # Denoising filter: fast Gaussian + median hybrid filter approximating Wiener residual
        denoised = cv2.GaussianBlur(gray_f, (5, 5), 1.0)
        median_denoised = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
        filtered = 0.5 * denoised + 0.5 * median_denoised

        # Residual noise pattern
        residual = gray_f - filtered
        return residual

    def analyze_sensor_integrity(self, image: np.ndarray) -> dict:
        """
        Analyze image for natural sensor pattern noise and block-wise consistency.
        
        Returns:
            dict containing:
                - prnu_energy: float (energy of noise residual; AI images typically have unnaturally low or anomalous uniform noise)
                - prnu_consistency_score: float (0.0 to 1.0; 1.0 means consistent natural sensor noise, <0.4 indicates AI or manipulation)
                - is_synthetic_noise_signature: bool (true if PRNU energy or variance distribution matches generative AI)
                - localized_noise_variance_std: float (high variance indicates localized tampering/pasting)
                - noise_map_rgb: np.ndarray (visualized residual noise map)
        """
        residual = self.extract_noise_residual(image)
        h, w = residual.shape

        # Overall residual energy
        prnu_energy = float(np.mean(residual ** 2))
        global_std = float(np.std(residual))

        # Block-wise analysis to check for localized tampering (e.g. pasted signatures or text)
        block_variances = []
        bs = self.block_size
        for y in range(0, h - bs + 1, bs):
            for x in range(0, w - bs + 1, bs):
                block = residual[y:y+bs, x:x+bs]
                block_variances.append(float(np.var(block)))

        if len(block_variances) > 0:
            var_std = float(np.std(block_variances))
            var_mean = float(np.mean(block_variances) + 1e-6)
            # Relative inconsistency across image blocks
            inconsistency_ratio = var_std / var_mean
        else:
            var_std = 0.0
            inconsistency_ratio = 0.0

        # AI-generated media (Midjourney, DALL-E, Sora, Stable Diffusion) typically either
        # exhibits very low natural sensor noise (<0.8) or strange periodic noise spikes.
        # Authentic camera images typically have PRNU energy between 1.5 and 15.0 with low block variance inconsistency.
        is_synthetic = prnu_energy < 0.65 or prnu_energy > 40.0 or inconsistency_ratio > 2.8

        # Consistency score (higher is more consistent/authentic, lower is anomalous)
        consistency_score = 1.0 / (1.0 + np.exp(inconsistency_ratio - 1.5))
        if prnu_energy < 0.65:
            consistency_score *= 0.5

        # Visual representation of the noise residual
        normalized_residual = np.clip((residual - np.mean(residual)) / (3 * global_std + 1e-6) * 127 + 128, 0, 255).astype(np.uint8)
        colored_noise = cv2.applyColorMap(normalized_residual, cv2.COLORMAP_VIRIDIS)
        colored_noise_rgb = cv2.cvtColor(colored_noise, cv2.COLOR_BGR2RGB)

        return {
            "prnu_energy": round(prnu_energy, 4),
            "sensor_noise_std": round(global_std, 4),
            "block_inconsistency_ratio": round(inconsistency_ratio, 3),
            "prnu_consistency_score": round(float(consistency_score), 4),
            "is_synthetic_noise_signature": bool(is_synthetic),
            "noise_map_rgb": colored_noise_rgb,
        }
