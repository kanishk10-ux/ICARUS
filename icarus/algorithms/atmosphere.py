# -*- coding: utf-8 -*-
"""Atmosphere and Weather Scientific Algorithms Layer.

Provides verified, deterministic algorithms for meteorological wind vector decomposition,
precipitation statistics, temperature horizontal gradients, and atmospheric anomalies.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("icarus.algorithms.atmosphere")


class AtmosphereAlgorithms:
    """Core mathematical algorithms for atmospheric dynamics and weather analysis."""

    @staticmethod
    def compute_wind_vectors(
        u_mps: np.ndarray,
        v_mps: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculates wind speed (m/s) and meteorological direction (degrees from North).

        Meteorological direction represents the direction FROM which wind is blowing:
        dir = (270 - atan2(v, u) * 180 / pi) % 360
        """
        u = np.asarray(u_mps, dtype=np.float64)
        v = np.asarray(v_mps, dtype=np.float64)
        if u.shape != v.shape:
            raise ValueError(f"Shape mismatch: U {u.shape} != V {v.shape}")

        speed = np.sqrt(u ** 2 + v ** 2)
        # Mathematical angle in radians
        math_angle_deg = np.rad2deg(np.arctan2(v, u))
        # Meteorological direction
        meteo_dir = (270.0 - math_angle_deg) % 360.0

        valid_speed = speed[np.isfinite(speed)]
        valid_dir = meteo_dir[np.isfinite(meteo_dir)]

        if valid_speed.size == 0:
            return {"error": "All wind components are NaN or invalid."}

        mean_speed = float(np.mean(valid_speed))
        max_speed = float(np.max(valid_speed))

        # Vector average direction
        mean_u = float(np.mean(u[np.isfinite(u)]))
        mean_v = float(np.mean(v[np.isfinite(v)]))
        mean_vector_dir = (270.0 - np.rad2deg(np.arctan2(mean_v, mean_u))) % 360.0

        # Beaufort gale force check (speed >= 17.2 m/s / ~34 knots)
        gale_fraction = float(np.sum(valid_speed >= 17.2) / valid_speed.size)

        return {
            "mean_wind_speed_mps": round(mean_speed, 3),
            "max_wind_speed_mps": round(max_speed, 3),
            "min_wind_speed_mps": round(float(np.min(valid_speed)), 3),
            "mean_wind_direction_deg": round(mean_vector_dir, 1),
            "gale_force_fraction": round(gale_fraction, 4),
            "total_pixels_evaluated": int(valid_speed.size),
            "unit_speed": "m/s",
            "unit_direction": "degrees_from_north",
        }

    @staticmethod
    def analyze_precipitation(
        precip_mm_per_hour: np.ndarray,
        duration_hours: float = 1.0,
    ) -> Dict[str, Any]:
        """Calculates rainfall accumulation, intensity categories, and dry pixel fractions.

        WMO Intensity thresholds (mm/hr):
        Light: < 2.5
        Moderate: 2.5 - 10.0
        Heavy: 10.0 - 50.0
        Violent: >= 50.0
        """
        arr = np.asarray(precip_mm_per_hour, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return {"error": "All precipitation data are NaN or invalid."}

        # Non-negative precipitation
        valid_clean = np.maximum(0.0, valid)
        total_pixels = int(valid_clean.size)

        rain_pixels = valid_clean[valid_clean > 0.1]
        rain_fraction = float(rain_pixels.size / total_pixels) if total_pixels > 0 else 0.0

        mean_intensity = float(np.mean(rain_pixels)) if rain_pixels.size > 0 else 0.0
        max_intensity = float(np.max(valid_clean))

        return {
            "mean_rainfall_rate_mmph": round(float(np.mean(valid_clean)), 3),
            "mean_rain_area_rate_mmph": round(mean_intensity, 3),
            "max_rainfall_rate_mmph": round(max_intensity, 3),
            "total_areal_accumulation_mm": round(float(np.mean(valid_clean) * duration_hours), 3),
            "precipitation_coverage_fraction": round(rain_fraction, 4),
            "heavy_rain_fraction": round(float(np.sum(valid_clean >= 10.0) / total_pixels), 4),
            "total_pixels_evaluated": total_pixels,
            "unit_rate": "mm/h",
            "unit_accumulation": "mm",
        }

    @staticmethod
    def compute_temperature_gradient(
        temp_celsius: np.ndarray,
        dx_km: float = 10.0,
        dy_km: float = 10.0,
    ) -> Dict[str, Any]:
        """Calculates spatial temperature gradients (frontogenesis indicator in °C/100km)."""
        temp = np.asarray(temp_celsius, dtype=np.float64)
        valid = temp[np.isfinite(temp)]
        if valid.size == 0:
            return {"error": "All temperature data are NaN or invalid."}

        mean_temp = float(np.mean(valid))
        std_temp = float(np.std(valid))

        if temp.ndim == 2 and temp.shape[0] > 2 and temp.shape[1] > 2:
            dt_dx = np.gradient(temp, dx_km, axis=1)
            dt_dy = np.gradient(temp, dy_km, axis=0)
            grad_mag = np.sqrt(dt_dx ** 2 + dt_dy ** 2) * 100.0  # °C per 100 km
            valid_grad = grad_mag[np.isfinite(grad_mag)]
            mean_grad = round(float(np.mean(valid_grad)), 4)
            max_grad = round(float(np.max(valid_grad)), 4)
        else:
            mean_grad = None
            max_grad = None

        return {
            "mean_temperature_celsius": round(mean_temp, 3),
            "min_temperature_celsius": round(float(np.min(valid)), 3),
            "max_temperature_celsius": round(float(np.max(valid)), 3),
            "std_temperature_celsius": round(std_temp, 3),
            "mean_gradient_celsius_per_100km": mean_grad,
            "max_gradient_celsius_per_100km": max_grad,
            "total_pixels_evaluated": int(valid.size),
            "unit_temp": "°C",
            "unit_gradient": "°C/100km",
        }
