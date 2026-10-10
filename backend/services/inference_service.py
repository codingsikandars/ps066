"""
Real-Time Inference and Oceanographic What-If Simulation Engine.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import torch
from backend.core.state import state
from backend.services.ocean_service import ocean_service


class InferenceService:
    def run_custom_inference(
        self,
        base_day_idx: int = 0,
        sst_delta_degC: float = 0.0,
        sss_delta_psu: float = 0.0,
        ssh_delta_m: float = 0.0,
        wind_scale_factor: float = 1.0,
        current_scale_factor: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Applies physical atmospheric/oceanic perturbations to a real satellite snapshot
        and runs the deep learning model to reconstruct the 3D subsurface temperature response.
        """
        if not state.is_ready:
            state.load_all()

        base_day_idx = max(0, min(base_day_idx, state.test_inputs.shape[0] - 1))
        x_orig = state.test_inputs[base_day_idx:base_day_idx+1].copy()  # (1, 7, 101, 241)
        x_perturbed = x_orig.copy()

        # Apply perturbations
        x_perturbed[:, 0] += sst_delta_degC  # SST
        x_perturbed[:, 1] += sss_delta_psu   # SSS
        x_perturbed[:, 2] += ssh_delta_m     # SSH
        x_perturbed[:, 3] *= current_scale_factor  # U current
        x_perturbed[:, 4] *= current_scale_factor  # V current
        x_perturbed[:, 5] *= wind_scale_factor     # U wind
        x_perturbed[:, 6] *= wind_scale_factor     # V wind

        with torch.no_grad():
            t_orig = state.model(torch.from_numpy(x_orig).float())[0].cpu().numpy()  # (15, 101, 241)
            t_sim = state.model(torch.from_numpy(x_perturbed).float())[0].cpu().numpy()

        diff_3d = t_sim - t_orig

        # Compute Ocean Heat Content (OHC) change in upper 300m (rho * Cp * integral dT dz)
        # Approximate Cp = 3985 J/(kg C), rho = 1025 kg/m^3 -> rho*Cp = 4.084e6 J/(m^3 C)
        rho_cp = 4.084e6
        dz = np.diff(state.depths[:11])  # up to 200-300m
        
        # Calculate mean temperature delta per depth level
        mean_delta_per_depth = []
        for d_idx, d_m in enumerate(state.depths):
            masked_diff = diff_3d[d_idx][state.mask]
            mean_delta_per_depth.append({
                "depth_m": float(d_m),
                "mean_delta_degC": round(float(np.mean(masked_diff)), 3),
                "max_delta_degC": round(float(np.max(masked_diff)), 3),
                "min_delta_degC": round(float(np.min(masked_diff)), 3),
            })

        # Surface warming penetration depth: depth at which delta T falls below 10% of surface delta
        penetration_depth_m = 0.0
        if abs(sst_delta_degC) > 0.01:
            s_delta = abs(mean_delta_per_depth[0]["mean_delta_degC"])
            for item in mean_delta_per_depth:
                if abs(item["mean_delta_degC"]) >= 0.1 * s_delta:
                    penetration_depth_m = item["depth_m"]

        return {
            "parameters": {
                "base_day_idx": base_day_idx,
                "sst_delta_degC": sst_delta_degC,
                "sss_delta_psu": sss_delta_psu,
                "ssh_delta_m": ssh_delta_m,
                "wind_scale_factor": wind_scale_factor,
                "current_scale_factor": current_scale_factor,
            },
            "subsurface_impact": {
                "mean_surface_delta_degC": round(float(mean_delta_per_depth[0]["mean_delta_degC"]), 3),
                "mean_thermocline_delta_degC": round(float(mean_delta_per_depth[7]["mean_delta_degC"]), 3),  # 100m
                "mean_deep_delta_degC": round(float(mean_delta_per_depth[14]["mean_delta_degC"]), 3),  # 1000m
                "thermal_penetration_depth_m": penetration_depth_m,
            },
            "depth_profile_response": mean_delta_per_depth,
            "sample_profiles_at_key_stations": [
                {
                    "station": "Arabian Sea Upwelling (14°N, 58°E)",
                    "lat": 14.0, "lon": 58.0,
                    "depths": [float(d) for d in state.depths],
                    "baseline_temp": [round(float(v), 2) for v in t_orig[:, 36, 52]],
                    "simulated_temp": [round(float(v), 2) for v in t_sim[:, 36, 52]],
                },
                {
                    "station": "Bay of Bengal Warm Pool (15°N, 88°E)",
                    "lat": 15.0, "lon": 88.0,
                    "depths": [float(d) for d in state.depths],
                    "baseline_temp": [round(float(v), 2) for v in t_orig[:, 40, 172]],
                    "simulated_temp": [round(float(v), 2) for v in t_sim[:, 40, 172]],
                },
                {
                    "station": "Equatorial Indian Ocean (6°N, 80°E)",
                    "lat": 6.0, "lon": 80.0,
                    "depths": [float(d) for d in state.depths],
                    "baseline_temp": [round(float(v), 2) for v in t_orig[:, 4, 140]],
                    "simulated_temp": [round(float(v), 2) for v in t_sim[:, 4, 140]],
                }
            ]
        }


inference_service = InferenceService()
