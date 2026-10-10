"""
Live Deep Learning Inference & Scenario Simulation API routes.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from backend.services.inference_service import inference_service

router = APIRouter(prefix="/api", tags=["Inference & Simulation"])


class SimulationRequest(BaseModel):
    base_day_idx: int = Field(0, ge=0, le=100)
    sst_delta_degC: float = Field(0.0, ge=-5.0, le=5.0, description="SST perturbation in °C")
    sss_delta_psu: float = Field(0.0, ge=-5.0, le=5.0, description="SSS perturbation in PSU")
    ssh_delta_m: float = Field(0.0, ge=-1.0, le=1.0, description="SSH perturbation in meters")
    wind_scale_factor: float = Field(1.0, ge=0.0, le=3.0, description="Wind stress multiplier")
    current_scale_factor: float = Field(1.0, ge=0.0, le=3.0, description="Current velocity multiplier")


@router.post("/predict")
def run_prediction(req: SimulationRequest):
    return inference_service.run_custom_inference(
        base_day_idx=req.base_day_idx,
        sst_delta_degC=req.sst_delta_degC,
        sss_delta_psu=req.sss_delta_psu,
        ssh_delta_m=req.ssh_delta_m,
        wind_scale_factor=req.wind_scale_factor,
        current_scale_factor=req.current_scale_factor,
    )
