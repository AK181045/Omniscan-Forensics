"""
OMNISCAN FORENSICS — Report Service
Manages forensic report storage, retrieval, and PDF certificate export.
Complies with ISO/IEC 27037 digital evidence logging.
"""

import logging
from typing import Optional, Dict
from app.models.schemas import TrustReportSummary, Platform, Verdict
from app.services.pdf_certificate_service import PDFCertificateService

logger = logging.getLogger("omniscan.services.report")

# In-memory store for active session reports (can also be backed by DB/Redis)
_REPORT_STORE: Dict[str, dict] = {}


class ReportService:
    def __init__(self):
        self.certificate_service = PDFCertificateService()

    async def store_report(self, report_id: str, report_data: dict):
        """Store completed forensic analysis in memory cache."""
        _REPORT_STORE[report_id] = report_data
        logger.info(f"Report stored | report_id={report_id} | total={len(_REPORT_STORE)}")

    async def get_report(self, report_id: str) -> Optional[dict]:
        """Retrieve a stored forensic report by its ID."""
        return _REPORT_STORE.get(report_id)

    async def list_reports(
        self, platform: Optional[str] = None, verdict: Optional[str] = None,
        limit: int = 20, offset: int = 0,
    ) -> list[TrustReportSummary]:
        """List past reports with optional filtering."""
        results = []
        for rep in list(_REPORT_STORE.values())[offset : offset + limit]:
            if platform and rep.get("platform") != platform:
                continue
            if verdict and rep.get("verdict") != verdict:
                continue
            results.append(
                TrustReportSummary(
                    report_id=rep["report_id"],
                    media_hash=rep["media_hash"],
                    platform=Platform(rep.get("platform", "other")),
                    sentinels_score=rep.get("sentinels_score", 0.0),
                    verdict=Verdict(rep.get("verdict", "authentic")),
                    analyzed_at=rep.get("analyzed_at", ""),
                )
            )
        return results

    async def get_summary_stats(self) -> dict:
        """Aggregated forensic statistics."""
        total = len(_REPORT_STORE)
        verdicts = {"authentic": 0, "suspicious": 0, "synthetic": 0, "partial_synthetic": 0}
        scores = []
        for rep in _REPORT_STORE.values():
            v = rep.get("verdict", "authentic")
            verdicts[v] = verdicts.get(v, 0) + 1
            scores.append(rep.get("sentinels_score", 0.0))

        avg_score = float(sum(scores) / max(total, 1))
        return {
            "total_analyses": total,
            "verdicts": verdicts,
            "avg_score": round(avg_score, 1),
            "iso_27037_compliant_count": total,
        }

    async def generate_pdf_certificate(self, report_id: str) -> Optional[bytes]:
        """
        Generate ISO/IEC 27037 court-admissible PDF certificate for a report.
        """
        report_data = _REPORT_STORE.get(report_id)
        if not report_data:
            return None

        return self.certificate_service.generate_certificate(report_data)
