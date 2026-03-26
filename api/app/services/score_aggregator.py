"""
SENTINELS OF INTEGRITY — Score Aggregator
Combines ML + Blockchain results into Sentinels Score (0-100).
"""

import logging
from app.models.schemas import MLResultResponse, BlockchainResultResponse, Verdict

logger = logging.getLogger("sentinels.services.score_aggregator")

WEIGHTS = {
    "ml_confidence": 0.50,
    "spatial_analysis": 0.15,
    "temporal_analysis": 0.15,
    "frequency_analysis": 0.10,
    "blockchain_provenance": 0.10,
}

THRESHOLD_AUTHENTIC = 70.0
THRESHOLD_SUSPICIOUS = 40.0


class ScoreAggregator:
    def compute_score(
        self, ml_result: MLResultResponse, blockchain_result: BlockchainResultResponse
    ) -> tuple[float, Verdict]:
        ml_trust = (1.0 - ml_result.confidence) * 100

        spatial_score = 50.0
        if ml_result.spatial:
            spatial_score = ml_result.spatial.face_consistency_score * 80 + (1 - ml_result.spatial.edge_artifact_score) * 20

        temporal_score = 50.0
        if ml_result.temporal:
            temporal_score = ml_result.temporal.temporal_consistency * 100

        frequency_score = 50.0
        if ml_result.frequency:
            frequency_score = (1 - ml_result.frequency.spectral_anomaly_score) * 100
            if ml_result.frequency.ghosting_detected:
                frequency_score *= 0.5

        blockchain_score = 40.0
        if blockchain_result.is_registered:
            blockchain_score = 95.0
            if blockchain_result.zk_verified:
                blockchain_score = 100.0

        final_score = (
            WEIGHTS["ml_confidence"] * ml_trust
            + WEIGHTS["spatial_analysis"] * spatial_score
            + WEIGHTS["temporal_analysis"] * temporal_score
            + WEIGHTS["frequency_analysis"] * frequency_score
            + WEIGHTS["blockchain_provenance"] * blockchain_score
        )
        final_score = max(0.0, min(100.0, final_score))

        # Partial Synthetic Detection Logic
        has_anomalies = False
        if ml_result.temporal and ml_result.temporal.anomaly_frames:
            # If there are anomaly frames but the overall temporal score is relatively high
            # it indicates only a portion of the video was altered.
            has_anomalies = len(ml_result.temporal.anomaly_frames) > 0 and len(ml_result.temporal.anomaly_frames) < (ml_result.temporal.frames_analyzed * 0.5)

        is_partial = getattr(ml_result, "is_partial", False)
        if is_partial:
            has_anomalies = True

        # Weight adjustments based on core analysis
        if ml_result.is_synthetic:
            final_score = min(final_score, 35.0)
        elif is_partial:
            final_score = min(max(final_score, 45.0), 65.0)

        if final_score >= THRESHOLD_AUTHENTIC:
            if has_anomalies:
                verdict = Verdict.PARTIAL_SYNTHETIC
            else:
                verdict = Verdict.AUTHENTIC
        elif final_score >= THRESHOLD_SUSPICIOUS:
            if has_anomalies or ("edge_bleeding" in ml_result.artifacts) or ("background_warping" in ml_result.artifacts):
                verdict = Verdict.PARTIAL_SYNTHETIC
            else:
                verdict = Verdict.SUSPICIOUS
        else:
            if is_partial:
                verdict = Verdict.PARTIAL_SYNTHETIC
            else:
                verdict = Verdict.SYNTHETIC

        logger.info(f"Score: {final_score:.1f} | Verdict: {verdict.value}")
        return round(final_score, 1), verdict.value
