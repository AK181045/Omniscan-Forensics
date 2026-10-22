"""
=============================================================================
SENTINELS OF INTEGRITY — Reports Route
GET /api/v1/reports — Retrieve and list Trust Reports.
=============================================================================
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from app.models.schemas import TrustReportSummary
from app.services.report_service import ReportService
from app.middleware.auth import get_current_user

logger = logging.getLogger("sentinels.api.reports")

router = APIRouter()


@router.get("/reports", response_model=list[TrustReportSummary])
async def list_reports(
    platform: Optional[str] = Query(None, description="Filter by platform"),
    verdict: Optional[str] = Query(None, description="Filter by verdict"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    user: dict = Depends(get_current_user),
):
    """
    List recent Trust Reports with optional filtering.

    Returns summaries of past analyses, paginated and optionally
    filtered by platform or verdict.
    """
    report_service = ReportService()
    reports = await report_service.list_reports(
        platform=platform,
        verdict=verdict,
        limit=limit,
        offset=offset,
    )
    return reports


@router.get("/reports/{report_id}")
async def get_report(
    report_id: str,
    user: dict = Depends(get_current_user),
):
    """Retrieve a full Trust Report by its ID."""
    report_service = ReportService()
    report = await report_service.get_report(report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    return report


@router.get("/reports/{report_id}/certificate")
async def download_certificate(
    report_id: str,
):
    """
    Export an ISO/IEC 27037 compliant court-admissible forensic PDF certificate.
    """
    from fastapi.responses import Response
    report_service = ReportService()
    pdf_bytes = await report_service.generate_pdf_certificate(report_id)
    if not pdf_bytes:
        raise HTTPException(status_code=404, detail="Forensic report or certificate not found")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="OmniScan_Forensic_Certificate_{report_id[:8]}.pdf"',
            "Content-Type": "application/pdf",
        },
    )


@router.get("/reports/stats/summary")
async def get_stats_summary(
    user: dict = Depends(get_current_user),
):
    """
    Get aggregated statistics across all analyses.

    Returns counts of authentic, suspicious, and synthetic verdicts,
    average scores, and platform distribution.
    """
    report_service = ReportService()
    stats = await report_service.get_summary_stats()
    return stats
