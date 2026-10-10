"""
Oceanographic data & spatial slicing API routes.
"""

from fastapi import APIRouter, Query
from backend.services.ocean_service import ocean_service

router = APIRouter(prefix="/api", tags=["Ocean Exploration"])


@router.get("/overview")
def get_overview():
    return ocean_service.get_overview()


@router.get("/slice")
def get_slice(
    day_idx: int = Query(0, ge=0, le=100, description="Test snapshot index"),
    variable: str = Query("temperature", description="Variable name ('temperature', 'sst', 'sss', 'ssh', 'u_curr', 'v_curr', 'u_wind', 'v_wind')"),
    depth_m: float = Query(0.0, description="Depth level in meters (for 3D temperature)"),
    mode: str = Query("pred", description="Display mode: 'pred', 'target', or 'error'"),
):
    return ocean_service.get_slice(day_idx=day_idx, variable_type=variable, depth_m=depth_m, mode=mode)


@router.get("/profile")
def get_profile(
    day_idx: int = Query(0, ge=0, le=100),
    lat: float = Query(15.0, ge=5.0, le=30.0),
    lon: float = Query(70.0, ge=45.0, le=105.0),
):
    return ocean_service.get_profile(day_idx=day_idx, lat=lat, lon=lon)


@router.get("/transect")
def get_transect(
    day_idx: int = Query(0, ge=0, le=100),
    axis: str = Query("lat", regex="^(lat|lon)$", description="'lat' for zonal (constant latitude), 'lon' for meridional"),
    coord_val: float = Query(15.0, description="Latitude (if axis=lat) or Longitude (if axis=lon)"),
):
    return ocean_service.get_transect(day_idx=day_idx, axis=axis, coord_val=coord_val)
