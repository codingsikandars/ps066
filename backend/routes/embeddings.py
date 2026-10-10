"""
Latent Satellite Embeddings & Attention Maps API routes.
"""

from fastapi import APIRouter, Query
from backend.services.embedding_service import embedding_service

router = APIRouter(prefix="/api", tags=["Satellite Embeddings"])


@router.get("/embeddings")
def get_embeddings(
    day_idx: int = Query(0, ge=0, le=100, description="Snapshot day index"),
):
    return embedding_service.extract_latent_embeddings(day_idx=day_idx)
