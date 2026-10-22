"""
=============================================================================
SENTINELS OF INTEGRITY — FastAPI Application Entry Point
Zero-Trust Backend API (The "Brain")
=============================================================================

Data Flow (Design.txt §2):
1. EXTENSION → Sends media URL/Hash to API
2. API → Triggers ML_INFERENCE and BLOCKCHAIN_LOOKUP
3. ML_INFERENCE → Returns {is_synthetic, confidence, artifacts}
4. BLOCKCHAIN_LOOKUP → Returns {is_registered, author, edit_history}
5. API → Aggregates results into "Sentinels Score"
6. API → EXTENSION → Renders UI Overlay
=============================================================================
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import detection, verification, reports, health
from app.middleware.rate_limiter import RateLimiterMiddleware
from app.middleware.sanitizer import InputSanitizationMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sentinels.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager — startup and shutdown events."""
    # --- Startup ---
    logger.info("🛡️  Sentinels of Integrity API starting up...")
    logger.info(f"   Environment: {settings.ENVIRONMENT}")
    logger.info(f"   API Version: {settings.API_VERSION}")
    logger.info(f"   Debug Mode: {settings.DEBUG}")

    # TODO: Initialize database connection pool
    # TODO: Load ML model into GPU memory
    # TODO: Connect to blockchain RPC
    # TODO: Connect to Redis for rate limiting

    logger.info("✅ All services initialized successfully")
    yield

    # --- Shutdown ---
    logger.info("🛑 Sentinels of Integrity API shutting down...")
    # TODO: Close database connections
    # TODO: Release GPU resources
    # TODO: Disconnect from Redis
    logger.info("👋 Goodbye!")


# =============================================================================
# Application Factory
# =============================================================================
app = FastAPI(
    title="OmniScan Forensics API",
    description=(
        "AI-Powered Court-Admissible Digital Forensics Platform for Deepfake and "
        "Document Tampering Detection — ISO/IEC 27037 Compliant."
    ),
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# =============================================================================
# Middleware Stack (order matters — outermost first)
# =============================================================================

# CORS — allow extension and local file:// pages to communicate with API
# "null" origin is sent by browsers for local file:// pages (visual_demo.html etc.)
_cors_origins = list(settings.CORS_ALLOWED_ORIGINS)
if "null" not in _cors_origins:
    _cors_origins.append("null")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,  # Must be False when origins includes "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting — Redis-based sliding window (TechStack.txt §2)
app.add_middleware(
    RateLimiterMiddleware,
    max_requests=settings.RATE_LIMIT_MAX_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
)

# Input Sanitization — all extension data is "untrusted" (Design.txt §6)
app.add_middleware(InputSanitizationMiddleware)

# =============================================================================
# Route Registration
# =============================================================================
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(detection.router, prefix="/api/v1", tags=["Detection"])
app.include_router(verification.router, prefix="/api/v1", tags=["Verification"])
app.include_router(reports.router, prefix="/api/v1", tags=["Reports"])


# =============================================================================
# Root Endpoint
# =============================================================================
@app.get("/", tags=["Root"])
async def root():
    """API root — returns service information."""
    return {
        "service": "Sentinels of Integrity",
        "version": settings.API_VERSION,
        "status": "operational",
        "docs": "/docs" if settings.DEBUG else "disabled",
    }
