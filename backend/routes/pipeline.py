"""
Data Harmonization Pipeline routes.
"""

from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(prefix="/api/pipeline", tags=["Data Harmonization Pipeline"])


@router.get("/spec")
def get_pipeline_spec() -> Dict[str, Any]:
    return {
        "title": "OceanEmbed Multi-Source Satellite Harmonization Pipeline",
        "spatial_grid": {
            "lat_bounds": [5.0, 30.0],
            "lon_bounds": [45.0, 105.0],
            "target_resolution_deg": 0.25,
            "dimensions": [101, 241],
            "total_points": 24341,
            "ocean_points": 11742,
        },
        "vertical_levels_m": [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
        "sources": [
            {
                "variable": "SST",
                "product": "OSTIA / Copernicus Marine (MOI-00168)",
                "native_res": "0.05° Daily",
                "interpolation": "Bilinear Spatial Downsampling to 0.25°",
                "status": "Ready",
            },
            {
                "variable": "SSS",
                "product": "SMAP / SMOS Multi-Observation (MOI-00051)",
                "native_res": "0.125° Daily",
                "interpolation": "Bilinear Spatial Downsampling to 0.25°",
                "status": "Ready",
            },
            {
                "variable": "SSH / SLA",
                "product": "DUACS Altimetry (MOI-00145)",
                "native_res": "0.25° Daily",
                "interpolation": "Target Grid Alignment & Land-Mask Clipping",
                "status": "Ready",
            },
            {
                "variable": "Currents (U, V)",
                "product": "OSCAR Surface Currents Final V2.0",
                "native_res": "0.25° Daily",
                "interpolation": "Vector Coordinate Alignment & Coastline Fill",
                "status": "Ready",
            },
            {
                "variable": "Winds (U, V)",
                "product": "ASCAT / CCMP V3.1 10m Winds",
                "native_res": "0.25° 6-hourly",
                "interpolation": "Daily Temporal Averaging & Grid Alignment",
                "status": "Ready",
            },
            {
                "variable": "Subsurface Target",
                "product": "GLORYS12V1 Ocean Physics Reanalysis (MOI-00021)",
                "native_res": "0.083° Daily, 50 vertical levels",
                "interpolation": "0.25° Horizontal Bilinear + 15-Level Vertical Spline",
                "status": "Ready",
            },
        ],
        "steps": [
            {"step": 1, "name": "Multi-Source Product Ingestion", "desc": "Automated download & verification from Copernicus Marine & PO.DAAC"},
            {"step": 2, "name": "Temporal & Spatial Regridding", "desc": "Standardize all 7 surface feeds to daily 0.25° NIO bounding box [5-30N, 45-105E]"},
            {"step": 3, "name": "Physical Land-Sea Masking", "desc": "Apply high-resolution coastal mask (11,742 water points)"},
            {"step": 4, "name": "Feature Normalization & CoordConv", "desc": "Z-score normalization and 2D positional coordinate injection (Lat/Lon channels)"},
            {"step": 5, "name": "Embeddings & 3D Reconstruction", "desc": "Pass through Hybrid ViT/ResNet encoder and depth-attention decoder"},
        ]
    }
