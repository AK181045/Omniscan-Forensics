"""
OMNISCAN FORENSICS — Error Level Analysis (ELA) Engine
Reference: N. Krawetz (2007) - 'Error Level Analysis for Digital Image Forensics'
Identifies localized compression history differentials, pasted elements,
forged signatures, stamp anomalies, and altered text in documents and images.
"""

import cv2
import numpy as np
import io
from PIL import Image, ImageChops, ImageEnhance
import logging

logger = logging.getLogger("omniscan.forensics.ela")


class ELAAnalyzer:
    """
    Error Level Analysis (ELA) works by intentionally re-saving an image at
    a known compression level (e.g. 90-95%) and calculating the pixel-by-pixel
    difference against the original matrix.
    
    Unmodified regions reach an error equilibrium, whereas spliced or pasted
    regions (such as forged signatures or manipulated text) exhibit higher error levels.
    """

    def __init__(self, quality: int = 92, scale: int = 15):
        self.quality = quality
        self.scale = scale

    def analyze_matrix(self, image_bgr: np.ndarray) -> dict:
        """
        Analyze a raw BGR image numpy matrix.
        
        Returns:
            dict containing:
                - ela_score: float (0.0 to 1.0, higher means greater tampering likelihood)
                - max_diff: float
                - mean_diff: float
                - ela_anomaly_detected: bool
                - localized_clusters: int
                - ela_image_rgb: np.ndarray (visualized amplified error map)
        """
        # Convert BGR to PIL RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_orig = Image.fromarray(image_rgb)

        # Re-compress in memory to JPEG
        buffer = io.BytesIO()
        pil_orig.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        pil_recompressed = Image.open(buffer)

        # Compute difference
        diff = ImageChops.difference(pil_orig, pil_recompressed)
        
        # Scale difference for visualization & quantitative thresholding
        diff_arr = np.asarray(diff, dtype=np.float32)
        diff_mag = np.max(diff_arr, axis=2)  # max channel difference per pixel

        # Extrema & statistical properties
        mean_diff = float(np.mean(diff_mag))
        std_diff = float(np.std(diff_mag))
        max_diff = float(np.max(diff_mag))
        p95_diff = float(np.percentile(diff_mag, 95))

        # Check for localized high-error clusters (indicative of localized paste/tampering)
        # Threshold: areas significantly higher than background average
        local_threshold = mean_diff + 2.5 * std_diff
        tampered_mask = (diff_mag > max(local_threshold, 15.0)).astype(np.uint8)
        
        # Find connected components to isolate suspicious segments/signatures
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tampered_mask)
        # Filter clusters by minimum size (ignoring single-pixel noise)
        significant_clusters = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] > 40]
        cluster_count = len(significant_clusters)

        # Compute normalized ELA anomaly score [0.0 - 1.0]
        # In a natural image, mean diff is typically 1-6. Tampered regions push p95 and cluster count up.
        raw_score = (p95_diff / 35.0) * 0.5 + (min(cluster_count, 10) / 10.0) * 0.5
        ela_score = float(np.clip(raw_score, 0.0, 1.0))

        # Generate enhanced visualization array
        enhanced_diff = np.clip(diff_mag * self.scale, 0, 255).astype(np.uint8)
        ela_visual_colormap = cv2.applyColorMap(enhanced_diff, cv2.COLORMAP_JET)
        ela_visual_rgb = cv2.cvtColor(ela_visual_colormap, cv2.COLOR_BGR2RGB)

        return {
            "ela_score": round(ela_score, 4),
            "mean_error": round(mean_diff, 2),
            "p95_error": round(p95_diff, 2),
            "max_error": round(max_diff, 2),
            "cluster_count": cluster_count,
            "ela_anomaly_detected": ela_score > 0.45 or cluster_count >= 3,
            "ela_image_rgb": ela_visual_rgb,
        }

    def analyze_bytes(self, image_bytes: bytes) -> dict:
        """Analyze raw image bytes directly."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes into pixel matrix")
        return self.analyze_matrix(img)
