# -*- coding: utf-8 -*-
"""Planetary Science Scientific Algorithms Layer.

Provides verified, deterministic algorithms for Digital Elevation Model (DEM) terrain slope,
aspect, surface roughness, and impact crater morphometry (circularity, depth-to-diameter ratio)
independent of neural networks.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("icarus.algorithms.planetary")


class PlanetaryAlgorithms:
    """Core mathematical algorithms for planetary surface morphology, lunar topography, and crater analysis."""

    @staticmethod
    def compute_terrain_slope_and_aspect(
        elevation_dem_m: np.ndarray,
        pixel_resolution_m: float = 100.0,
    ) -> Dict[str, Any]:
        """Calculates terrain slope (degrees) and topographic aspect (azimuth) from a DEM grid.

        Slope = arctan(sqrt((dz/dx)^2 + (dz/dy)^2)) * 180 / pi
        """
        dem = np.asarray(elevation_dem_m, dtype=np.float64)
        valid = dem[np.isfinite(dem)]
        if valid.size == 0 or dem.ndim != 2:
            return {"error": "DEM must be a valid 2D array of finite elevations."}

        # Horn's finite difference gradient
        dz_dy, dz_dx = np.gradient(dem, pixel_resolution_m)

        gradient_mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
        slope_deg = np.rad2deg(np.arctan(gradient_mag))

        # Aspect: direction of maximum downhill slope (degrees clockwise from North)
        aspect_deg = (np.rad2deg(np.arctan2(-dz_dx, dz_dy)) + 360.0) % 360.0

        valid_slope = slope_deg[np.isfinite(slope_deg)]
        valid_aspect = aspect_deg[np.isfinite(aspect_deg)]

        # Roughness: standard deviation of slope
        roughness = float(np.std(valid_slope))

        return {
            "mean_slope_degrees": round(float(np.mean(valid_slope)), 2),
            "max_slope_degrees": round(float(np.max(valid_slope)), 2),
            "min_slope_degrees": round(float(np.min(valid_slope)), 2),
            "terrain_roughness_index": round(roughness, 3),
            "mean_elevation_m": round(float(np.mean(valid)), 2),
            "elevation_range_m": round(float(np.max(valid) - np.min(valid)), 2),
            "steep_slope_fraction": round(float(np.sum(valid_slope > 25.0) / valid_slope.size), 4),
            "total_pixels_evaluated": int(valid.size),
            "pixel_resolution_m": pixel_resolution_m,
            "unit_slope": "degrees",
            "unit_elevation": "meters",
        }

    @staticmethod
    def analyze_crater_morphometry(
        diameter_km: float,
        depth_km: float,
        rim_perimeter_km: Optional[float] = None,
        crater_area_km2: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluates impact crater structural morphology and simple vs. complex transition.

        Lunar transition diameter is ~15-20 km (Pike 1977):
        - Simple craters: bowl-shaped, depth/diameter ~ 0.2
        - Complex craters: central peak / flat floor, depth/diameter ~ 0.05 - 0.1
        """
        d = float(diameter_km)
        h = float(depth_km)

        if d <= 0 or h <= 0:
            return {"error": "Crater diameter and depth must be strictly positive."}

        depth_to_diameter = h / d

        # Simple vs complex classification
        if d < 15.0:
            morphology_class = "Simple crater (bowl-shaped)"
            expected_depth = 0.2 * d
        else:
            morphology_class = "Complex crater (central peak / terraced rim)"
            expected_depth = 1.044 * (d ** 0.301)  # Pike (1977) complex crater scaling

        # Circularity C = 4 * pi * Area / Perimeter^2 (1.0 for perfect circle)
        circularity = None
        if rim_perimeter_km and crater_area_km2:
            p = float(rim_perimeter_km)
            a = float(crater_area_km2)
            if p > 0:
                circularity = round(float(4.0 * np.pi * a / (p ** 2)), 4)

        return {
            "diameter_km": round(d, 3),
            "depth_km": round(h, 3),
            "depth_to_diameter_ratio": round(float(depth_to_diameter), 4),
            "morphology_class": morphology_class,
            "rim_circularity_index": circularity,
            "is_circular": circularity is not None and circularity >= 0.85,
            "expected_lunar_depth_km": round(float(expected_depth), 3),
            "depth_anomaly_ratio": round(float(h / expected_depth), 3),
        }
