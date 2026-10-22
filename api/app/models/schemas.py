"""
=============================================================================
SENTINELS OF INTEGRITY — Pydantic Schemas
Request/Response models for API endpoints.
=============================================================================
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, HttpUrl, field_validator
from app.middleware.sanitizer import sanitize_string


# =============================================================================
# Enums
# =============================================================================
class Platform(str, Enum):
    YOUTUBE = "youtube"
    TWITTER = "twitter"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    REDDIT = "reddit"
    OTHER = "other"
    DEVICE = "device"


class MediaType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"


class Verdict(str, Enum):
    AUTHENTIC = "authentic"
    SUSPICIOUS = "suspicious"
    SYNTHETIC = "synthetic"
    PARTIAL_SYNTHETIC = "partial_synthetic"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AIOriginSource(str, Enum):
    UNKNOWN_AI = "unknown_ai"
    HUMAN_CREATED = "human_created"


# =============================================================================
# AI Origin Detection Schemas
# =============================================================================
# Schemas removed.


# =============================================================================
# Request Schemas
# =============================================================================
class AnalysisOptions(BaseModel):
    """Optional analysis configuration."""
    skip_blockchain: bool = Field(default=False, description="Skip blockchain lookup")
    detailed_report: bool = Field(default=False, description="Include per-frame details")
    max_frames: int = Field(default=0, ge=0, le=300, description="Max frames (0=auto)")
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Custom confidence threshold for flagging",
    )


class MediaAnalysisRequest(BaseModel):
    """Request body for POST /detect."""
    media_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        pattern=r"^[a-fA-F0-9]{64}$",
        description="SHA-256 hex hash of the media (computed client-side)",
    )
    media_url: str = Field(
        ...,
        max_length=2000,
        description="URL of the media on the platform (or device:// URI)",
    )
    platform: Platform = Field(
        ...,
        description="Source platform",
    )
    media_type: MediaType = Field(
        default=MediaType.VIDEO,
        description="Type of media",
    )
    options: Optional[AnalysisOptions] = Field(
        default=None,
        description="Optional analysis configuration",
    )
    extension_version: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Extension version for compatibility",
    )

    @field_validator("media_hash")
    @classmethod
    def sanitize_hash(cls, v: str) -> str:
        return v.lower()  # Normalize to lowercase hex

    @field_validator("extension_version")
    @classmethod
    def sanitize_version(cls, v: Optional[str]) -> Optional[str]:
        if v:
            return sanitize_string(v)
        return v


# =============================================================================
# Response Schemas — ML Result
# =============================================================================
class SpatialAnalysis(BaseModel):
    face_consistency_score: float = Field(ge=0.0, le=1.0)
    edge_artifact_score: float = Field(ge=0.0, le=1.0)
    faces_detected: int = Field(ge=0)


class TemporalAnalysis(BaseModel):
    temporal_consistency: float = Field(ge=0.0, le=1.0)
    frames_analyzed: int = Field(ge=0)
    anomaly_frames: list[int] = Field(default_factory=list)


class FrequencyAnalysis(BaseModel):
    spectral_anomaly_score: float = Field(ge=0.0, le=1.0)
    ghosting_detected: bool = False
    grid_leakage_detected: bool = False
    dct_moments: dict = Field(default_factory=dict)


class DocumentForensicAnalysis(BaseModel):
    ela_score: float = Field(ge=0.0, le=1.0)
    ela_anomaly_detected: bool = False
    cluster_count: int = 0
    prnu_consistency: float = Field(ge=0.0, le=1.0)
    prnu_synthetic_flag: bool = False
    layout_integrity_score: float = Field(default=1.0, ge=0.0, le=1.0)


class VideoForensicAnalysis(BaseModel):
    motion_continuity_score: float = Field(ge=0.0, le=1.0)
    dropped_or_frozen_frames: int = 0
    optical_flow_jerk: float = 0.0
    grid_leakage_detected: bool = False
    dct_moments: dict = Field(default_factory=dict)


class EvidenceVisuals(BaseModel):
    grad_cam_base64: Optional[str] = None
    ela_base64: Optional[str] = None
    spectrum_base64: Optional[str] = None
    flow_base64: Optional[str] = None


class MLResultResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    is_synthetic: bool
    is_partial: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    artifacts: list[str] = Field(default_factory=list)
    model_version: str = "OmniScan-SRNet-F3Net-v5.0"
    spatial: Optional[SpatialAnalysis] = None
    temporal: Optional[TemporalAnalysis] = None
    frequency: Optional[FrequencyAnalysis] = None
    document_forensics: Optional[DocumentForensicAnalysis] = None
    video_forensics: Optional[VideoForensicAnalysis] = None
    visual_evidence: Optional[EvidenceVisuals] = None


# =============================================================================
# Response Schemas — Blockchain Result
# =============================================================================
class EditRecord(BaseModel):
    editor: str
    edit_hash: str
    timestamp: str
    description: str


class BlockchainResultResponse(BaseModel):
    is_registered: bool
    author: Optional[str] = None
    registration_tx: Optional[str] = None
    edit_history: list[EditRecord] = Field(default_factory=list)
    zk_verified: bool = False


class BlockchainVerificationResponse(BaseModel):
    """Response for GET /verify endpoint."""
    content_hash: str
    is_registered: bool
    author: Optional[str] = None
    registration_tx: Optional[str] = None
    edit_history: list[EditRecord] = Field(default_factory=list)
    zk_verified: bool = False
    lookup_latency_ms: float = 0.0


# =============================================================================
# Response Schemas — Full Analysis
# =============================================================================
class MediaAnalysisResponse(BaseModel):
    """Full analysis response for POST /detect."""
    report_id: str
    media_hash: str
    media_url: str
    platform: Platform
    sentinels_score: float = Field(ge=0.0, le=100.0)
    verdict: Verdict
    ml_result: MLResultResponse
    blockchain_result: BlockchainResultResponse
    status: AnalysisStatus
    analyzed_at: str
    forensic_modality: str = "video"
    certificate_url: Optional[str] = None
    iso_27037_compliant: bool = True


class TrustReportSummary(BaseModel):
    """Lightweight report summary for list views."""
    report_id: str
    media_hash: str
    platform: Platform
    sentinels_score: float
    verdict: Verdict
    analyzed_at: str
