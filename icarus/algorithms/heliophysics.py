# -*- coding: utf-8 -*-
"""Heliophysics and Space Weather Scientific Algorithms Layer.

Provides verified, deterministic algorithms for solar flare classification (GOES A/B/C/M/X scale),
geomagnetic storm categorization (NOAA G-scale from Kp and Dst index), and solar differential rotation.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("icarus.algorithms.heliophysics")


class HeliophysicsAlgorithms:
    """Core mathematical algorithms for solar physics, heliophysics, and space weather monitoring."""

    @staticmethod
    def classify_goes_solar_flare(peak_flux_watts_per_m2: float) -> Dict[str, Any]:
        """Classifies solar flare magnitude according to official NOAA / GOES X-ray flux scale.

        Scale boundaries (0.1 - 0.8 nm flux in W/m²):
        A: < 1e-7
        B: 1e-7 to 1e-6
        C: 1e-6 to 1e-5 (Minor)
        M: 1e-5 to 1e-4 (Moderate, potential HF radio blackout)
        X: >= 1e-4 (Extreme, severe radiation storm)
        """
        flux = float(peak_flux_watts_per_m2)
        if flux <= 0:
            return {"error": "Solar flux must be strictly positive."}

        if flux < 1e-7:
            flare_class = "A"
            numeric_scale = flux / 1e-8
            severity = 0
        elif flux < 1e-6:
            flare_class = "B"
            numeric_scale = flux / 1e-7
            severity = 1
        elif flux < 1e-5:
            flare_class = "C"
            numeric_scale = flux / 1e-6
            severity = 2
        elif flux < 1e-4:
            flare_class = "M"
            numeric_scale = flux / 1e-5
            severity = 3
        else:
            flare_class = "X"
            numeric_scale = flux / 1e-4
            severity = 4

        goes_code = f"{flare_class}{numeric_scale:.1f}"

        return {
            "goes_flare_class": goes_code,
            "flare_category": flare_class,
            "numeric_magnitude": round(float(numeric_scale), 2),
            "peak_flux_watts_per_m2": flux,
            "severity_index": severity,
            "potential_hf_blackout": severity >= 3,
            "unit_flux": "W/m²",
        }

    @staticmethod
    def evaluate_geomagnetic_storm(
        kp_index: float,
        dst_index_nt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Categorizes geomagnetic disturbance using planetary Kp index and storm-time Dst index.

        NOAA G-scale:
        Kp < 5: G0 (Quiet / Unsettled)
        Kp = 5: G1 (Minor)
        Kp = 6: G2 (Moderate)
        Kp = 7: G3 (Strong)
        Kp = 8: G4 (Severe)
        Kp = 9: G5 (Extreme)
        """
        kp = float(kp_index)
        if kp < 0 or kp > 9:
            raise ValueError(f"Kp index must be between 0.0 and 9.0, got {kp}")

        if kp >= 9.0:
            g_scale = "G5 - Extreme"
            storm_active = True
        elif kp >= 8.0:
            g_scale = "G4 - Severe"
            storm_active = True
        elif kp >= 7.0:
            g_scale = "G3 - Strong"
            storm_active = True
        elif kp >= 6.0:
            g_scale = "G2 - Moderate"
            storm_active = True
        elif kp >= 5.0:
            g_scale = "G1 - Minor"
            storm_active = True
        else:
            g_scale = "G0 - Below storm threshold"
            storm_active = False

        # Dst interpretation (nT)
        dst_status = "unknown"
        if dst_index_nt is not None:
            dst = float(dst_index_nt)
            if dst <= -200:
                dst_status = "Super-storm"
            elif dst <= -100:
                dst_status = "Intense storm"
            elif dst <= -50:
                dst_status = "Moderate storm"
            elif dst <= -30:
                dst_status = "Weak storm"
            else:
                dst_status = "Quiet / Normal"

        return {
            "kp_index": round(kp, 2),
            "noaa_g_scale": g_scale,
            "is_geomagnetic_storm": storm_active,
            "dst_index_nt": dst_index_nt,
            "dst_storm_status": dst_status,
        }

    @staticmethod
    def compute_solar_differential_rotation(latitude_deg: float) -> Dict[str, Any]:
        """Calculates solar surface angular velocity (degrees/day) via Snodgrass (1984) relation.

        omega(phi) = A + B * sin^2(phi) + C * sin^4(phi)
        A = 14.71 deg/day, B = -2.39 deg/day, C = -1.78 deg/day
        """
        phi_deg = float(latitude_deg)
        if abs(phi_deg) > 90.0:
            raise ValueError("Solar latitude must be between -90 and +90 degrees.")

        phi_rad = np.deg2rad(phi_deg)
        sin_phi = np.sin(phi_rad)

        omega = 14.71 - 2.39 * (sin_phi ** 2) - 1.78 * (sin_phi ** 4)
        rotation_period_days = 360.0 / omega

        return {
            "heliographic_latitude_deg": phi_deg,
            "angular_velocity_deg_per_day": round(float(omega), 4),
            "synodic_rotation_period_days": round(float(rotation_period_days), 2),
            "equatorial_period_days": 24.47,
            "polar_period_days": 34.16,
            "unit_velocity": "degrees/day",
            "unit_period": "days",
        }
