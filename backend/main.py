"""
FastAPI Main Application for OceanEmbed Platform.
SIH Problem Statement 26066.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.core.state import state
from backend.services.ocean_service import ocean_service
from backend.routes.ocean import router as ocean_router
from backend.routes.embeddings import router as embeddings_router
from backend.routes.inference import router as inference_router
from backend.routes.validation import router as validation_router
from backend.routes.pipeline import router as pipeline_router

STATIC_DIR = ROOT_DIR / "backend" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models, data arrays, ARGO floats, and precompute predictions
    print("[OceanEmbed API] Starting up server and loading weights/data...")
    state.load_all()
    ocean_service.ensure_predictions()
    print("[OceanEmbed API] Server initialization completed successfully!")
    yield
    print("[OceanEmbed API] Server shutting down...")


app = FastAPI(
    title="OceanEmbed — Subsurface Ocean Temperature Reconstruction API",
    description="SIH Problem Statement 26066: Multi-modal satellite embedding framework for 3D ocean temperature reconstruction over the North Indian Ocean.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(ocean_router)
app.include_router(embeddings_router)
app.include_router(inference_router)
app.include_router(validation_router)
app.include_router(pipeline_router)

# Mount Static Assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": state.is_ready,
        "device": "CPU",
        "domain": "North Indian Ocean (5°N-30°N, 45°E-105°E)",
    }


@app.get("/")
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "OceanEmbed API is running. Access /docs for Swagger UI."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
