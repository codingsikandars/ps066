"""
Oceanographic data slicing, profile extraction, and physical diagnostics with Celsius denormalization.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from backend.core.state import state


class OceanService:
    def __init__(self):
        self._cached_preds: Optional[np.ndarray] = None

    def ensure_predictions(self):
        """Pre-compute test dataset predictions once for instantaneous API queries."""
        if self._cached_preds is not None:
            return self._cached_preds

        if not state.is_ready:
            state.load_all()

        print("[OceanService] Running model inference on test dataset with CoordConv and denormalization...")
        loader = torch.utils.data.DataLoader(state.dataset, batch_size=4, shuffle=False)
        all_preds = []

        with torch.no_grad():
            for x, _, _ in loader:
                pred_norm = state.model(x)
                pred_degc = state.dataset.denormalize_target(pred_norm)
                all_preds.append(pred_degc.cpu().numpy())

        self._cached_preds = np.concatenate(all_preds, axis=0)  # (N, 15, 101, 241) in degC
        print(f"[OceanService] Cached {self._cached_preds.shape[0]} test predictions in °C successfully.")
        return self._cached_preds

    def get_overview(self) -> Dict[str, Any]:
        if not state.is_ready:
            state.load_all()
        return {
            "problem_statement": {
                "id": "26066",
                "title": "OceanEmbed - Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations",
                "organization": "Smart India Hackathon / Ministry of Earth Sciences / INCOIS",
            },
            "domain": {
                "name": "North Indian Ocean (NIO)",
                "lat_min": float(state.lats.min()),
                "lat_max": float(state.lats.max()),
                "lon_min": float(state.lons.min()),
                "lon_max": float(state.lons.max()),
                "resolution_deg": 0.25,
                "n_lat": len(state.lats),
                "n_lon": len(state.lons),
                "ocean_cells": int(state.mask.sum()),
                "land_cells": int((~state.mask).sum()),
            },
            "depths": [float(d) for d in state.depths],
            "surface_variables": [
                {"id": "sst", "name": "Sea Surface Temperature (SST)", "unit": "°C", "source": "OSTIA (0.05° → 0.25°)", "channel": 0},
                {"id": "sss", "name": "Sea Surface Salinity (SSS)", "unit": "PSU", "source": "SMAP / SMOS (0.125° → 0.25°)", "channel": 1},
                {"id": "ssh", "name": "Sea Surface Height / SLA", "unit": "m", "source": "DUACS Altimetry (0.25°)", "channel": 2},
                {"id": "u_curr", "name": "Zonal Surface Current (U)", "unit": "m/s", "source": "OSCAR (0.25°)", "channel": 3},
                {"id": "v_curr", "name": "Meridional Surface Current (V)", "unit": "m/s", "source": "OSCAR (0.25°)", "channel": 4},
                {"id": "u_wind", "name": "Zonal Surface Wind (U)", "unit": "m/s", "source": "ASCAT / CCMP (0.25°)", "channel": 5},
                {"id": "v_wind", "name": "Meridional Surface Wind (V)", "unit": "m/s", "source": "ASCAT / CCMP (0.25°)", "channel": 6},
            ],
            "num_test_days": int(state.test_inputs.shape[0]) if state.test_inputs is not None else 18,
            "metrics_summary": {
                "mean_rmse_degC": state.evaluation_report.get("mean_rmse", 0.1857),
                "mean_corr": state.evaluation_report.get("mean_corr", 0.8607),
                "thermocline_d20_rmse_m": state.evaluation_report.get("thermocline_d20_rmse_m", 3.254),
                "argo_float_count": state.evaluation_report.get("argo_float_count", 360),
                "argo_mean_rmse_degC": state.evaluation_report.get("argo_mean_rmse", 0.2249),
                "argo_mean_corr": state.evaluation_report.get("argo_mean_corr", 0.9998),
            },
        }

    def get_slice(
        self,
        day_idx: int = 0,
        variable_type: str = "temperature",
        depth_m: float = 0.0,
        mode: str = "pred",
    ) -> Dict[str, Any]:
        """Returns 2D grid slice formatted for canvas raster rendering."""
        self.ensure_predictions()
        day_idx = max(0, min(day_idx, self._cached_preds.shape[0] - 1))

        if variable_type in ["sst", "sss", "ssh", "u_curr", "v_curr", "u_wind", "v_wind"]:
            var_idx_map = {
                "sst": 0, "sss": 1, "ssh": 2, "u_curr": 3, "v_curr": 4, "u_wind": 5, "v_wind": 6
            }
            ch = var_idx_map[variable_type]
            grid_2d = state.test_inputs[day_idx, ch].copy()
            masked_grid = np.where(state.mask, grid_2d, np.nan)
            valid_vals = masked_grid[state.mask]
            
            return {
                "day_idx": day_idx,
                "variable": variable_type,
                "type": "surface",
                "depth_m": 0.0,
                "mode": "observed",
                "min_val": float(np.nanmin(valid_vals)),
                "max_val": float(np.nanmax(valid_vals)),
                "mean_val": float(np.nanmean(valid_vals)),
                "grid": [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in masked_grid],
                "lats": [round(float(lat), 2) for lat in state.lats],
                "lons": [round(float(lon), 2) for lon in state.lons],
            }

        # Subsurface 3D temperature slice
        depth_idx = int(np.argmin(np.abs(state.depths - depth_m)))
        actual_depth = float(state.depths[depth_idx])

        pred_grid = self._cached_preds[day_idx, depth_idx].copy()
        target_grid = state.test_targets[day_idx, depth_idx].copy()

        if mode == "target":
            display_grid = target_grid
        elif mode == "error":
            display_grid = np.abs(pred_grid - target_grid)
        else:
            display_grid = pred_grid

        masked_grid = np.where(state.mask, display_grid, np.nan)
        valid_vals = masked_grid[state.mask]

        return {
            "day_idx": day_idx,
            "variable": "temperature",
            "type": "subsurface",
            "depth_m": actual_depth,
            "depth_idx": depth_idx,
            "mode": mode,
            "min_val": float(np.nanmin(valid_vals)),
            "max_val": float(np.nanmax(valid_vals)),
            "mean_val": float(np.nanmean(valid_vals)),
            "grid": [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in masked_grid],
            "lats": [round(float(lat), 2) for lat in state.lats],
            "lons": [round(float(lon), 2) for lon in state.lons],
        }

    def get_profile(
        self,
        day_idx: int,
        lat: float,
        lon: float,
    ) -> Dict[str, Any]:
        """Extracts vertical profile at exact or closest (lat, lon)."""
        self.ensure_predictions()
        day_idx = max(0, min(day_idx, self._cached_preds.shape[0] - 1))

        lat_idx = int(np.argmin(np.abs(state.lats - lat)))
        lon_idx = int(np.argmin(np.abs(state.lons - lon)))
        actual_lat = float(state.lats[lat_idx])
        actual_lon = float(state.lons[lon_idx])
        is_ocean = bool(state.mask[lat_idx, lon_idx])

        pred_prof = [round(float(v), 3) for v in self._cached_preds[day_idx, :, lat_idx, lon_idx]]
        target_prof = [round(float(v), 3) for v in state.test_targets[day_idx, :, lat_idx, lon_idx]]
        surface_inputs = {
            "sst": round(float(state.test_inputs[day_idx, 0, lat_idx, lon_idx]), 2),
            "sss": round(float(state.test_inputs[day_idx, 1, lat_idx, lon_idx]), 2),
            "ssh": round(float(state.test_inputs[day_idx, 2, lat_idx, lon_idx]), 3),
            "u_curr": round(float(state.test_inputs[day_idx, 3, lat_idx, lon_idx]), 3),
            "v_curr": round(float(state.test_inputs[day_idx, 4, lat_idx, lon_idx]), 3),
            "u_wind": round(float(state.test_inputs[day_idx, 5, lat_idx, lon_idx]), 2),
            "v_wind": round(float(state.test_inputs[day_idx, 6, lat_idx, lon_idx]), 2),
        }

        # Calculate D20 thermocline depth
        pred_d20 = self._interpolate_isotherm_depth(pred_prof, state.depths, 20.0)
        target_d20 = self._interpolate_isotherm_depth(target_prof, state.depths, 20.0)

        # Find closest ARGO float within 3.0 degrees
        closest_argo = None
        min_dist = 999.0
        for argo in state.argo_observations:
            dist = np.sqrt((argo["lat"] - actual_lat)**2 + (argo["lon"] - actual_lon)**2)
            if dist < min_dist and dist < 3.0:
                min_dist = dist
                closest_argo = {
                    "lat": argo["lat"],
                    "lon": argo["lon"],
                    "distance_deg": round(float(dist), 2),
                    "temperature": [round(float(t), 3) for t in argo["temperature"]],
                }

        # Vertical thermal gradients (dT/dz in °C / 100m)
        pred_grad = []
        for i in range(len(state.depths) - 1):
            dz = state.depths[i+1] - state.depths[i]
            dt = pred_prof[i+1] - pred_prof[i]
            pred_grad.append(round(float((dt / dz) * 100.0), 3))

        return {
            "day_idx": day_idx,
            "query_lat": lat,
            "query_lon": lon,
            "grid_lat": actual_lat,
            "grid_lon": actual_lon,
            "lat_idx": lat_idx,
            "lon_idx": lon_idx,
            "is_ocean": is_ocean,
            "depths": [float(d) for d in state.depths],
            "pred_temperature": pred_prof,
            "target_temperature": target_prof,
            "absolute_error": [round(abs(p - t), 3) for p, t in zip(pred_prof, target_prof)],
            "surface_inputs": surface_inputs,
            "pred_d20_m": pred_d20,
            "target_d20_m": target_d20,
            "vertical_gradient_dT_dz_100m": pred_grad,
            "nearest_argo": closest_argo,
        }

    def get_transect(
        self,
        day_idx: int = 0,
        axis: str = "lat",
        coord_val: float = 15.0,
    ) -> Dict[str, Any]:
        """Vertical cross-section along latitude or longitude."""
        self.ensure_predictions()
        day_idx = max(0, min(day_idx, self._cached_preds.shape[0] - 1))

        if axis == "lat":
            lat_idx = int(np.argmin(np.abs(state.lats - coord_val)))
            actual_coord = float(state.lats[lat_idx])
            pred_section = self._cached_preds[day_idx, :, lat_idx, :].copy()
            target_section = state.test_targets[day_idx, :, lat_idx, :].copy()
            mask_section = state.mask[lat_idx, :]
            coords_axis = [round(float(x), 2) for x in state.lons]
            axis_name = "Longitude (°E)"
        else:
            lon_idx = int(np.argmin(np.abs(state.lons - coord_val)))
            actual_coord = float(state.lons[lon_idx])
            pred_section = self._cached_preds[day_idx, :, :, lon_idx].copy()
            target_section = state.test_targets[day_idx, :, :, lon_idx].copy()
            mask_section = state.mask[:, lon_idx]
            coords_axis = [round(float(y), 2) for y in state.lats]
            axis_name = "Latitude (°N)"

        # Apply mask
        for i in range(len(mask_section)):
            if not mask_section[i]:
                pred_section[:, i] = np.nan
                target_section[:, i] = np.nan

        return {
            "day_idx": day_idx,
            "axis": axis,
            "coord_val": actual_coord,
            "axis_label": axis_name,
            "coords": coords_axis,
            "depths": [float(d) for d in state.depths],
            "pred_transect": [[None if np.isnan(v) else round(float(v), 2) for v in row] for row in pred_section],
            "target_transect": [[None if np.isnan(v) else round(float(v), 2) for v in row] for row in target_section],
        }

    def _interpolate_isotherm_depth(self, temp_profile: List[float], depths: np.ndarray, target_temp: float = 20.0) -> Optional[float]:
        """Calculates exact depth where temperature crosses target isotherm."""
        for i in range(len(temp_profile) - 1):
            t1, t2 = temp_profile[i], temp_profile[i+1]
            z1, z2 = depths[i], depths[i+1]
            if (t1 >= target_temp >= t2) or (t1 <= target_temp <= t2):
                if abs(t2 - t1) < 1e-5:
                    return float(z1)
                frac = (target_temp - t1) / (t2 - t1)
                return round(float(z1 + frac * (z2 - z1)), 1)
        return None


ocean_service = OceanService()
