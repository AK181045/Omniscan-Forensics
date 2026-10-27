"""
=============================================================================
SENTINELS OF INTEGRITY — Integration Test: ML Pipeline
Tests the full inference pipeline: frames → model → score.
=============================================================================
"""

import pytest
import os
import tempfile
import numpy as np
import cv2
from ml_models.preprocessing.frame_extractor import FrameExtractor
from ml_models.preprocessing.face_detector import FaceDetector
from ml_models.preprocessing.frequency_analyzer import FrequencyAnalyzer
from ml_models.inference.trust_scorer import TrustScorer


class TestMLPipelineIntegration:
    """End-to-end integration test for the ML pipeline."""

    @pytest.fixture
    def sample_video(self):
        """Create a synthetic test video."""
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_name = tmp.name
        tmp.close()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_name, fourcc, 30.0, (640, 480))
        for i in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (200, 150), (440, 330), (200, 180, 160), -1)
            writer.write(frame)
        writer.release()
        yield tmp_name
        try:
            os.unlink(tmp_name)
        except Exception:
            pass

    def test_full_pipeline_frame_extraction(self, sample_video):
        """Frames should be extracted and resized to 299x299."""
        extractor = FrameExtractor(target_size=(299, 299), max_frames=30)
        frames = extractor.extract_from_file(sample_video)
        assert frames.shape[1:] == (299, 299, 3)
        assert len(frames) > 0

    def test_full_pipeline_face_detection(self, sample_video):
        """Face detector should process frames without errors."""
        extractor = FrameExtractor(max_frames=10)
        frames = extractor.extract_from_file(sample_video)
        detector = FaceDetector()
        processed = detector.process_video_frames(frames)
        assert processed.shape[1:] == (299, 299, 3)

    def test_full_pipeline_frequency_analysis(self, sample_video):
        """Frequency analyzer should return valid scores."""
        extractor = FrameExtractor(max_frames=10)
        frames = extractor.extract_from_file(sample_video)
        analyzer = FrequencyAnalyzer()
        result = analyzer.analyze_video(frames)
        assert 0.0 <= result["spectral_anomaly_score"] <= 1.0
        assert isinstance(result["ghosting_detected"], bool)

    def test_full_pipeline_trust_score(self, sample_video):
        """Trust scorer should produce valid output from mock ML results."""
        scorer = TrustScorer()
        ml_result = {"confidence": 0.3, "artifacts": []}
        freq_result = {"ghosting_detected": False, "spectral_anomaly_score": 0.15}
        report = scorer.compute(ml_result, freq_result)
        assert "is_synthetic" in report
        assert "confidence" in report
        assert "model_version" in report

    def test_pipeline_combined(self, sample_video):
        """Full combined pipeline should work end to end."""
        # 1. Extract frames
        extractor = FrameExtractor(max_frames=10)
        frames = extractor.extract_from_file(sample_video)

        # 2. Frequency analysis
        freq_analyzer = FrequencyAnalyzer()
        freq_result = freq_analyzer.analyze_video(frames)

        # 3. Trust score (with mock ML)
        scorer = TrustScorer()
        ml_mock = {"confidence": 0.15, "artifacts": []}
        report = scorer.compute(ml_mock, freq_result)

        assert report["is_synthetic"] is False
        assert report["model_version"] == "xception-gru-v1"
