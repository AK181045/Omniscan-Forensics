"""
OMNISCAN FORENSICS — Dual-Domain (2D-FFT / 2D-DCT / DWT) Spectral Analyzer
References:
1. G. R. Suryawanshi & S. N. Mali (2018) - 'Analysis of Effect of Spatial Domain
   Steganography Technique on DCT Domain using Statistical Features'
2. OmniScan Architecture (2D-FFT/DCT grid leakages & GAN/Diffusion upscaling artifacts)

Extracts frequency-domain representations and higher-order statistical moments:
- Mean (μ), Standard Deviation (σ), Skewness (S), Kurtosis (K) on AC/DC DCT coefficients.
- 2D-FFT Azimuthal Profile & High-Frequency Peak Detection (periodic grid artifacts).
- DWT (Discrete Wavelet Transform - Haar) subband energy distribution (LL, LH, HL, HH).
"""

import cv2
import numpy as np
import scipy.stats
import scipy.fft
import pywt
import logging

logger = logging.getLogger("omniscan.forensics.frequency")


class FrequencyAnalyzer:
    """
    Advanced Transform-Domain Analyzer for Universal Steganalysis and AI artifact isolation.
    """

    def __init__(self, high_freq_threshold: float = 0.35):
        self.high_freq_threshold = high_freq_threshold

    def compute_fft(self, frame: np.ndarray) -> np.ndarray:
        """Compute 2D FFT magnitude spectrum (log scale) of a grayscale frame."""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(frame.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))
        return magnitude

    def compute_dct_blockwise(self, frame: np.ndarray, block_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute block-wise 8x8 2D-DCT coefficients across the entire image.
        Returns:
            dc_coeffs: 1D array of DC coefficients
            ac_coeffs: 1D array of non-zero AC coefficients
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        h, w = gray.shape
        # Crop to multiple of block_size
        h_crop = (h // block_size) * block_size
        w_crop = (w // block_size) * block_size
        gray_crop = gray[:h_crop, :w_crop].astype(np.float32) - 128.0

        dc_list = []
        ac_list = []

        for y in range(0, h_crop, block_size):
            for x in range(0, w_crop, block_size):
                block = gray_crop[y:y+block_size, x:x+block_size]
                dct_block = cv2.dct(block)
                dc_list.append(dct_block[0, 0])
                # AC coefficients are everything except [0,0]
                ac_vals = dct_block.flatten()[1:]
                ac_list.extend(ac_vals)

        return np.array(dc_list, dtype=np.float32), np.array(ac_list, dtype=np.float32)

    def compute_suryawanshi_mali_moments(self, frame: np.ndarray) -> dict:
        """
        Calculates higher-order statistical moments on AC & DC DCT coefficients
        as established in Suryawanshi & Mali (2018):
        - Mean (μ)
        - Standard Deviation (σ)
        - Skewness (S)
        - Kurtosis (K)
        """
        dc_coeffs, ac_coeffs = self.compute_dct_blockwise(frame)

        def moments(arr):
            if len(arr) == 0:
                return 0.0, 0.0, 0.0, 0.0
            u = float(np.mean(arr))
            s = float(np.std(arr))
            skew = float(scipy.stats.skew(arr)) if s > 1e-6 else 0.0
            kurt = float(scipy.stats.kurtosis(arr)) if s > 1e-6 else 0.0
            return round(u, 3), round(s, 3), round(skew, 3), round(kurt, 3)

        dc_u, dc_s, dc_skew, dc_kurt = moments(dc_coeffs)
        ac_u, ac_s, ac_skew, ac_kurt = moments(ac_coeffs)

        return {
            "dct_dc_mean": dc_u,
            "dct_dc_std": dc_s,
            "dct_dc_skewness": dc_skew,
            "dct_dc_kurtosis": dc_kurt,
            "dct_ac_mean": ac_u,
            "dct_ac_std": ac_s,
            "dct_ac_skewness": ac_skew,
            "dct_ac_kurtosis": ac_kurt,
        }

    def detect_spectral_grid_leakage(self, frame: np.ndarray) -> tuple[bool, float, dict]:
        """
        Detects periodic grid artifacts and high-frequency spectral leakage
        typical of generative AI upsampling convolutions (GANs / Diffusion Models).
        """
        spectrum = self.compute_fft(frame)
        h, w = spectrum.shape
        cy, cx = h // 2, w // 2

        # Radial mask for high frequencies
        r_inner = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y - cy)**2 + (x - cx)**2)
        high_freq_mask = dist_from_center > r_inner

        high_energy = float(np.mean(spectrum[high_freq_mask]))
        total_energy = float(np.mean(spectrum) + 1e-8)
        energy_ratio = high_energy / total_energy

        # Peak detection in high-frequency quadrant (spikes indicative of regular lattice/grid)
        hf_spectrum = spectrum * high_freq_mask
        hf_max = float(np.max(hf_spectrum))
        hf_mean = float(np.mean(hf_spectrum[high_freq_mask]))
        hf_peak_ratio = hf_max / (hf_mean + 1e-6)

        # Periodic lattice flag: high peak-to-mean ratio in outer bands
        grid_detected = hf_peak_ratio > 3.2 or energy_ratio > self.high_freq_threshold
        anomaly_score = float(np.clip((energy_ratio - 0.2) / 0.3 * 0.5 + (hf_peak_ratio / 6.0) * 0.5, 0.0, 1.0))

        details = {
            "energy_ratio": round(energy_ratio, 4),
            "hf_peak_to_mean": round(hf_peak_ratio, 3),
            "grid_artifact_detected": grid_detected,
        }
        return grid_detected, round(anomaly_score, 4), details

    def compute_dwt_haar_bands(self, frame: np.ndarray) -> dict:
        """
        Discrete Wavelet Transform (Haar) 2D decomposition into 4 subbands:
        - LL: Low-frequency approximation
        - LH: Horizontal details
        - HL: Vertical details
        - HH: High-frequency diagonal noise
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        coeffs = pywt.dwt2(gray.astype(np.float32), 'haar')
        LL, (LH, HL, HH) = coeffs

        energy_ll = float(np.sum(LL ** 2))
        energy_lh = float(np.sum(LH ** 2))
        energy_hl = float(np.sum(HL ** 2))
        energy_hh = float(np.sum(HH ** 2))
        total = energy_ll + energy_lh + energy_hl + energy_hh + 1e-6

        hh_ratio = energy_hh / total
        detail_ratio = (energy_lh + energy_hl + energy_hh) / total

        return {
            "dwt_hh_energy_ratio": round(hh_ratio, 5),
            "dwt_detail_energy_ratio": round(detail_ratio, 5),
            "dwt_ll_energy_ratio": round(energy_ll / total, 5),
        }

    def analyze_frame(self, frame: np.ndarray) -> dict:
        """Complete transform-domain forensic analysis of a single frame or image."""
        grid_detected, anomaly_score, grid_details = self.detect_spectral_grid_leakage(frame)
        moments = self.compute_suryawanshi_mali_moments(frame)
        dwt_bands = self.compute_dwt_haar_bands(frame)

        # Normalized spectral representation for visualization
        spectrum = self.compute_fft(frame)
        norm_spec = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_spec = cv2.applyColorMap(norm_spec, cv2.COLORMAP_MAGMA)
        colored_spec_rgb = cv2.cvtColor(colored_spec, cv2.COLOR_BGR2RGB)

        return {
            "spectral_anomaly_score": anomaly_score,
            "grid_artifacts_present": grid_detected,
            "ghosting_detected": grid_detected,
            "fft_metrics": grid_details,
            "dct_moments": moments,
            "dwt_analysis": dwt_bands,
            "spectrum_image_rgb": colored_spec_rgb,
        }

    def analyze_video(self, frames: np.ndarray) -> dict:
        """Aggregate frequency analysis across a sequence of video frames."""
        scores = [self.analyze_frame(f) for f in frames]
        avg_anomaly = np.mean([s["spectral_anomaly_score"] for s in scores])
        grid_count = sum(1 for s in scores if s["grid_artifacts_present"])
        sample_frame_res = scores[0] if len(scores) > 0 else {}

        return {
            "spectral_anomaly_score": round(float(avg_anomaly), 4),
            "grid_artifacts_present": grid_count > len(frames) * 0.25,
            "ghosting_detected": grid_count > len(frames) * 0.25,
            "grid_frame_ratio": round(grid_count / max(len(frames), 1), 3),
            "dct_moments": sample_frame_res.get("dct_moments", {}),
            "dwt_analysis": sample_frame_res.get("dwt_analysis", {}),
            "spectrum_image_rgb": sample_frame_res.get("spectrum_image_rgb", None),
        }
