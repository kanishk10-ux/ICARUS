# -*- coding: utf-8 -*-
"""Oceans and Water Scientific Algorithms Layer.

Provides verified, deterministic algorithms for Sea Surface Temperature (SST) anomaly detection,
marine heatwave thresholding, salinity statistics, chlorophyll-a optical estimation,
and ocean current velocity and vorticity calculations.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("icarus.algorithms.oceans")


class OceanAlgorithms:
    """Core mathematical algorithms for oceanography and marine environmental analysis."""

    @staticmethod
    def analyze_sst_anomalies(
        sst_observed_celsius: np.ndarray,
        sst_climatology_celsius: np.ndarray,
        mhw_threshold_celsius: float = 1.5,
    ) -> Dict[str, Any]:
        """Calculates sea surface temperature anomalies and identifies marine heatwave conditions."""
        obs = np.asarray(sst_observed_celsius, dtype=np.float64)
        clim = np.asarray(sst_climatology_celsius, dtype=np.float64)
        if obs.shape != clim.shape:
            raise ValueError(f"Shape mismatch: Observed {obs.shape} != Climatology {clim.shape}")

        anomaly = obs - clim
        valid = anomaly[np.isfinite(anomaly)]
        if valid.size == 0:
            return {"error": "All SST anomaly pixels are NaN or masked."}

        mhw_mask = valid >= mhw_threshold_celsius
        total_valid = int(valid.size)
        mhw_count = int(np.sum(mhw_mask))

        return {
            "mean_anomaly_celsius": round(float(np.mean(valid)), 4),
            "std_anomaly_celsius": round(float(np.std(valid)), 4),
            "max_anomaly_celsius": round(float(np.max(valid)), 4),
            "min_anomaly_celsius": round(float(np.min(valid)), 4),
            "marine_heatwave_fraction": round(float(mhw_count / total_valid), 4),
            "mhw_threshold_applied": mhw_threshold_celsius,
            "total_pixels_evaluated": total_valid,
            "unit": "°C",
        }

    @staticmethod
    def analyze_salinity(salinity_psu: np.ndarray) -> Dict[str, Any]:
        """Computes statistical moments for practical salinity (PSS-78).

        Typical open ocean range: 32 to 38 PSU.
        """
        arr = np.asarray(salinity_psu, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return {"error": "All salinity pixels are NaN or masked."}

        mean_sal = float(np.mean(valid))
        return {
            "mean_salinity_psu": round(mean_sal, 4),
            "std_salinity_psu": round(float(np.std(valid)), 4),
            "min_salinity_psu": round(float(np.min(valid)), 4),
            "max_salinity_psu": round(float(np.max(valid)), 4),
            "median_salinity_psu": round(float(np.median(valid)), 4),
            "freshwater_influence_fraction": round(float(np.sum(valid < 30.0) / valid.size), 4),
            "hypersaline_fraction": round(float(np.sum(valid > 38.0) / valid.size), 4),
            "total_pixels_evaluated": int(valid.size),
            "unit": "PSU",
        }

    @staticmethod
    def estimate_chlorophyll_oc4(
        rrs_blue: np.ndarray,
        rrs_green: np.ndarray,
    ) -> np.ndarray:
        """Estimates chlorophyll-a concentration (mg/m³) via standard ocean-color band ratio.

        Approximation using empirical polynomial formulation: log10(Chl) = a0 + a1*R + a2*R^2...
        """
        blue = np.asarray(rrs_blue, dtype=np.float64)
        green = np.asarray(rrs_green, dtype=np.float64)
        if blue.shape != green.shape:
            raise ValueError(f"Shape mismatch: Blue {blue.shape} != Green {green.shape}")

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(green > 0, blue / green, np.nan)
            r = np.log10(np.clip(ratio, 0.01, 100.0))
            # NASA OC4 standard coefficients
            log_chl = 0.3272 - 2.9940 * r + 2.7118 * (r ** 2) - 0.8244 * (r ** 3)
            chl = np.power(10.0, log_chl)

        return np.clip(chl, 0.01, 100.0)

    @staticmethod
    def compute_ocean_current_kinematics(
        u_velocity_mps: np.ndarray,
        v_velocity_mps: np.ndarray,
        dx_m: float = 1000.0,
        dy_m: float = 1000.0,
    ) -> Dict[str, Any]:
        """Calculates current velocity magnitude (m/s) and vertical component of relative vorticity (1/s).

        Vorticity zeta = dv/dx - du/dy.
        """
        u = np.asarray(u_velocity_mps, dtype=np.float64)
        v = np.asarray(v_velocity_mps, dtype=np.float64)
        if u.shape != v.shape:
            raise ValueError(f"Shape mismatch: U {u.shape} != V {v.shape}")

        speed = np.sqrt(u ** 2 + v ** 2)
        valid_speed = speed[np.isfinite(speed)]
        if valid_speed.size == 0:
            return {"error": "All current velocity pixels are NaN or masked."}

        # Vorticity computation using finite difference gradients
        if u.ndim == 2 and u.shape[0] > 2 and u.shape[1] > 2:
            dv_dx = np.gradient(v, dx_m, axis=1)
            du_dy = np.gradient(u, dy_m, axis=0)
            vorticity = dv_dx - du_dy
            valid_vort = vorticity[np.isfinite(vorticity)]
            mean_vort = round(float(np.mean(valid_vort)), 8)
            std_vort = round(float(np.std(valid_vort)), 8)
        else:
            mean_vort = None
            std_vort = None

        return {
            "mean_current_speed_mps": round(float(np.mean(valid_speed)), 4),
            "max_current_speed_mps": round(float(np.max(valid_speed)), 4),
            "min_current_speed_mps": round(float(np.min(valid_speed)), 4),
            "mean_vorticity_per_sec": mean_vort,
            "std_vorticity_per_sec": std_vort,
            "total_pixels_evaluated": int(valid_speed.size),
            "unit_speed": "m/s",
            "unit_vorticity": "s⁻¹",
        }
