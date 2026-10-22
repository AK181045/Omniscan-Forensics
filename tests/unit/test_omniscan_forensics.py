"""
OMNISCAN FORENSICS — Core Unit Tests
Tests:
1. Header-Agnostic Raw Bitstream Ingestion
2. 2D-FFT/DCT Spectral Decomposition & Statistical Moments (Suryawanshi & Mali 2018)
3. Error Level Analysis (Krawetz 2007)
4. PRNU Sensor Noise Pattern Extraction (Lukas et al. 2006)
5. Farneback Dense Optical Flow Temporal Continuity
6. SRNet + Grad-CAM Spatial Localization (Selvaraju et al. 2017)
7. ISO/IEC 27037 Forensic PDF Certificate Generation (ReportLab)
"""

import numpy as np
import cv2
import torch
import io

from api.app.services.header_agnostic_ingestion import HeaderAgnosticIngestionEngine
from api.app.services.pdf_certificate_service import PDFCertificateService
from ml_models.preprocessing.frequency_analyzer import FrequencyAnalyzer
from ml_models.preprocessing.ela_analyzer import ELAAnalyzer
from ml_models.preprocessing.prnu_extractor import PRNUExtractor
from ml_models.preprocessing.optical_flow_analyzer import OpticalFlowAnalyzer
from ml_models.models.srnet_model import OmniScanFusionNet, GradCAMExplainer


def test_header_agnostic_ingestion():
    engine = HeaderAgnosticIngestionEngine()
    # Create test JPEG bytes
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[20:80, 20:80] = 200
    _, buf = cv2.imencode(".jpg", test_img)
    raw_bytes = buf.tobytes()

    res = engine.ingest_bytes(raw_bytes)
    assert len(res.media_hash) == 64
    assert res.media_type == "image"
    assert res.frame_count >= 1
    assert res.raw_frames.shape[1:] == (256, 256, 3)


def test_frequency_analyzer_and_moments():
    analyzer = FrequencyAnalyzer()
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    # Add high frequency periodic pattern
    y, x = np.ogrid[:128, :128]
    img[...] = ((np.sin(x) + np.cos(y)) * 100 + 128).astype(np.uint8)[..., None]

    res = analyzer.analyze_frame(img)
    assert "spectral_anomaly_score" in res
    assert "dct_moments" in res
    moments = res["dct_moments"]
    assert "dct_dc_mean" in moments
    assert "dct_ac_mean" in moments
    assert "dct_ac_skewness" in moments
    assert "dct_ac_kurtosis" in moments
    assert res["spectrum_image_rgb"] is not None


def test_ela_analyzer():
    ela = ELAAnalyzer(quality=90)
    img_bgr = np.full((128, 128, 3), 180, dtype=np.uint8)
    # Splice a square with sharp contrasting compression history
    img_bgr[40:90, 40:90] = [20, 50, 220]

    res = ela.analyze_matrix(img_bgr)
    assert "ela_score" in res
    assert "mean_error" in res
    assert "cluster_count" in res
    assert res["ela_image_rgb"].shape == (128, 128, 3)


def test_prnu_extractor():
    prnu = PRNUExtractor()
    # Synthetic clean image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    res = prnu.analyze_sensor_integrity(img)
    assert "prnu_energy" in res
    assert "prnu_consistency_score" in res
    assert res["noise_map_rgb"].shape == (100, 100, 3)


def test_optical_flow_analyzer():
    of = OpticalFlowAnalyzer()
    f1 = np.zeros((100, 100, 3), dtype=np.uint8)
    f2 = np.zeros((100, 100, 3), dtype=np.uint8)
    # Simulate movement
    f1[20:60, 20:60] = 255
    f2[25:65, 25:65] = 255

    frames = np.stack([f1, f2], axis=0)
    res = of.analyze_sequence(frames)
    assert "motion_continuity_score" in res
    assert "dropped_or_frozen_frames" in res
    assert res["flow_visual_rgb"] is not None


def test_srnet_grad_cam():
    model = OmniScanFusionNet(num_classes=2)
    model.eval()
    explainer = GradCAMExplainer(model)

    inp = torch.rand(1, 3, 128, 128)
    heatmap = explainer.generate_heatmap(inp, target_class=1)
    assert heatmap.shape == (128, 128)
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0

    orig = np.zeros((128, 128, 3), dtype=np.uint8)
    overlay = explainer.overlay_heatmap(orig, heatmap)
    assert overlay.shape == (128, 128, 3)


def test_pdf_certificate_service():
    cert_service = PDFCertificateService()
    test_data = {
        "report_id": "TEST-REPORT-12345",
        "media_hash": "a" * 64,
        "sentinels_score": 94.5,
        "verdict": "authentic",
        "analyzed_at": "2026-10-22T20:00:00Z",
        "media_type": "dynamic video bitstream",
        "grad_cam_rgb": np.zeros((100, 100, 3), dtype=np.uint8),
        "spectrum_rgb": np.zeros((100, 100, 3), dtype=np.uint8),
        "diagnostics": {
            "spectral_anomaly": 0.05,
            "prnu_consistency": 0.92,
            "synthetic_prob": 0.08,
            "motion_continuity": 0.98,
        },
    }

    pdf_bytes = cert_service.generate_certificate(test_data)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
