"""
=============================================================================
SENTINELS OF INTEGRITY — Unit Test: Frame Extractor
Tests video frame extraction at 30fps, 299x299.
=============================================================================
"""

import pytest
import numpy as np
import os
import tempfile
import cv2

from ml_models.preprocessing.frame_extractor import FrameExtractor


class TestFrameExtractor:
    """Tests for FrameExtractor."""

    @pytest.fixture
    def extractor(self):
        return FrameExtractor(target_size=(299, 299), target_fps=30, max_frames=90)

    @pytest.fixture
    def sample_video_path(self):
        """Create a minimal synthetic test video."""
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_name = tmp.name
        tmp.close()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_name, fourcc, 30.0, (640, 480))

        # Write 60 frames (~2 seconds at 30fps)
        for i in range(60):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        yield tmp_name
        try:
            os.unlink(tmp_name)
        except Exception:
            pass

    # =========================================================================
    # Test: Frame Extraction from File
    # =========================================================================

    def test_extract_returns_numpy_array(self, extractor, sample_video_path):
        """extract_from_file should return a numpy array."""
        frames = extractor.extract_from_file(sample_video_path)
        assert isinstance(frames, np.ndarray)

    def test_extracted_frame_dimensions(self, extractor, sample_video_path):
        """Frames should be resized to 299x299 (Design.txt §3)."""
        frames = extractor.extract_from_file(sample_video_path)
        assert frames.shape[1] == 299  # Height
        assert frames.shape[2] == 299  # Width
        assert frames.shape[3] == 3    # RGB channels

    def test_extracted_frame_count(self, extractor, sample_video_path):
        """Should extract frames at the target fps rate."""
        frames = extractor.extract_from_file(sample_video_path)
        assert len(frames) > 0
        assert len(frames) <= 90  # max_frames limit

    def test_max_frames_limit(self, sample_video_path):
        """Frame count should not exceed max_frames."""
        extractor = FrameExtractor(max_frames=10)
        frames = extractor.extract_from_file(sample_video_path)
        assert len(frames) <= 10

    def test_invalid_video_raises_error(self, extractor):
        """Should raise ValueError for invalid video path."""
        with pytest.raises(ValueError, match="Cannot open video"):
            extractor.extract_from_file("/nonexistent/video.mp4")

    # =========================================================================
    # Test: Frame Extraction from Bytes
    # =========================================================================

    def test_extract_from_bytes(self, extractor, sample_video_path):
        """extract_from_bytes should work with video byte content."""
        with open(sample_video_path, "rb") as f:
            video_bytes = f.read()

        frames = extractor.extract_from_bytes(video_bytes)
        assert isinstance(frames, np.ndarray)
        assert len(frames) > 0

    # =========================================================================
    # Test: Normalization
    # =========================================================================

    def test_normalize_frames_range(self):
        """Normalized frames should be in [0, 1] range."""
        frames = np.random.randint(0, 255, (10, 299, 299, 3), dtype=np.uint8)
        normalized = FrameExtractor.normalize_frames(frames)

        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_normalize_preserves_shape(self):
        """Normalization should not change the shape."""
        frames = np.random.randint(0, 255, (5, 299, 299, 3), dtype=np.uint8)
        normalized = FrameExtractor.normalize_frames(frames)
        assert normalized.shape == frames.shape


class TestFrameExtractorConfig:
    """Test that FrameExtractor respects configuration."""

    def test_default_config_values(self):
        """Default values should match Design.txt §3 specs."""
        extractor = FrameExtractor()
        assert extractor.target_size == (299, 299)
        assert extractor.target_fps == 30
        assert extractor.max_frames == 300

    def test_custom_config(self):
        """Custom config should override defaults."""
        extractor = FrameExtractor(target_size=(224, 224), target_fps=15, max_frames=50)
        assert extractor.target_size == (224, 224)
        assert extractor.target_fps == 15
        assert extractor.max_frames == 50
