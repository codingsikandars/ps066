"""
Scientific Validation, Benchmarking & ARGO float API routes.
"""

from fastapi import APIRouter, Query
from backend.services.validation_service import validation_service

router = APIRouter(prefix="/api", tags=["Validation & ARGO Floats"])


@router.get("/metrics")
def get_metrics():
    return validation_service.get_metrics_breakdown()


@router.get("/argo")
def get_argo_floats(limit: int = Query(360, ge=1, le=1000)):
    return validation_service.get_argo_floats(limit=limit)
