"""
=============================================================================
OMNISCAN FORENSICS — Detection Route
POST /api/v1/detect — Submit media for multi-modal forensics analysis.
POST /api/v1/detect/upload — Header-agnostic direct file upload.
=============================================================================
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
from app.models.schemas import (
    MediaAnalysisRequest,
    MediaAnalysisResponse,
    AnalysisStatus,
    Platform,
    MediaType,
)
from app.services.detection_service import DetectionService
from app.services.score_aggregator import ScoreAggregator
from app.middleware.auth import get_current_user

logger = logging.getLogger("omniscan.api.detection")

router = APIRouter()


@router.post("/detect", response_model=MediaAnalysisResponse)
async def detect_media(
    request: MediaAnalysisRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """
    Submit media for deepfake detection, steganography analysis, and provenance verification.
    """
    report_id = str(uuid.uuid4())
    logger.info(
        f"Detection request received | report_id={report_id} | "
        f"platform={request.platform} | media_hash={request.media_hash[:16]}..."
    )

    try:
        detection_service = DetectionService()

        # Run multi-modal forensics and blockchain lookup in parallel
        ml_result, blockchain_result, forensic_ctx = await detection_service.analyze(
            media_hash=request.media_hash,
            media_url=str(request.media_url),
            platform=request.platform.value if hasattr(request.platform, "value") else str(request.platform),
            media_type=request.media_type.value if hasattr(request.media_type, "value") else str(request.media_type),
            options=request.options,
        )

        # Aggregate into Sentinels / OmniScan Trust Score
        aggregator = ScoreAggregator()
        sentinels_score, verdict = aggregator.compute_score(ml_result, blockchain_result)

        certificate_url = f"/api/v1/reports/{report_id}/certificate"
        now_iso = datetime.now(timezone.utc).isoformat()

        response = MediaAnalysisResponse(
            report_id=report_id,
            media_hash=request.media_hash,
            media_url=str(request.media_url),
            platform=request.platform,
            sentinels_score=sentinels_score,
            verdict=verdict,
            ml_result=ml_result,
            blockchain_result=blockchain_result,
            status=AnalysisStatus.COMPLETED,
            analyzed_at=now_iso,
            forensic_modality=forensic_ctx.get("media_type", "video"),
            certificate_url=certificate_url,
            iso_27037_compliant=True,
        )

        # Merge report data with forensic context for PDF certificate generation
        report_store_payload = response.model_dump()
        report_store_payload.update(forensic_ctx)

        # Store report asynchronously
        background_tasks.add_task(
            detection_service.store_report, report_id, report_store_payload
        )

        logger.info(
            f"Detection completed | report_id={report_id} | "
            f"score={sentinels_score:.1f} | verdict={verdict}"
        )
        return response

    except TimeoutError:
        logger.error(f"Detection timed out | report_id={report_id}")
        raise HTTPException(
            status_code=504,
            detail="Analysis timed out. The media may be too large for real-time processing.",
        )
    except Exception as e:
        logger.error(f"Detection failed | report_id={report_id} | error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred during forensic examination: {str(e)}",
        )


@router.post("/detect/upload", response_model=MediaAnalysisResponse)
async def detect_uploaded_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    platform: str = Form("device"),
    media_type: str = Form("video"),
    user: dict = Depends(get_current_user),
):
    """
    Header-agnostic direct media upload endpoint:
    Demuxes raw pixel matrices directly from input bitstreams, bypassing file headers and EXIF tags.
    """
    report_id = str(uuid.uuid4())
    logger.info(f"Direct file upload received | filename={file.filename} | report_id={report_id}")

    try:
        raw_bytes = await file.read()
        detection_service = DetectionService()

        # Compute immediate SHA-256 hash for ISO/IEC 27037 compliance
        media_hash = detection_service.ingestion_engine.compute_sha256(raw_bytes)
        virtual_url = f"device://{file.filename}"

        ml_result, blockchain_result, forensic_ctx = await detection_service.analyze(
            media_hash=media_hash,
            media_url=virtual_url,
            platform=platform,
            media_type=media_type,
            raw_bytes=raw_bytes,
        )

        aggregator = ScoreAggregator()
        sentinels_score, verdict = aggregator.compute_score(ml_result, blockchain_result)

        certificate_url = f"/api/v1/reports/{report_id}/certificate"
        now_iso = datetime.now(timezone.utc).isoformat()

        response = MediaAnalysisResponse(
            report_id=report_id,
            media_hash=media_hash,
            media_url=virtual_url,
            platform=Platform.DEVICE,
            sentinels_score=sentinels_score,
            verdict=verdict,
            ml_result=ml_result,
            blockchain_result=blockchain_result,
            status=AnalysisStatus.COMPLETED,
            analyzed_at=now_iso,
            forensic_modality=forensic_ctx.get("media_type", "video"),
            certificate_url=certificate_url,
            iso_27037_compliant=True,
        )

        report_store_payload = response.model_dump()
        report_store_payload.update(forensic_ctx)

        background_tasks.add_task(
            detection_service.store_report, report_id, report_store_payload
        )

        return response

    except Exception as e:
        logger.error(f"File upload detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Forensic upload analysis failed: {str(e)}",
        )


@router.get("/detect/{report_id}", response_model=MediaAnalysisResponse)
async def get_detection_result(
    report_id: str,
    user: dict = Depends(get_current_user),
):
    """Retrieve a previously generated detection report by ID."""
    from app.services.report_service import ReportService
    report_service = ReportService()
    report = await report_service.get_report(report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    return report
