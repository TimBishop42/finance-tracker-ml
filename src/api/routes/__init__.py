"""API Router initialization module.

This module combines all feature-specific routers into a single router
that can be included in the main FastAPI application.
"""

from fastapi import APIRouter
from .predict import router as predict_router
from .train import router as train_router

# Create main router
router = APIRouter()

# Include feature routers with their prefixes and tags
router.include_router(predict_router, prefix="/predict", tags=["predict"])
router.include_router(train_router, prefix="/train", tags=["train"]) 