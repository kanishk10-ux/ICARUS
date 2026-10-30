# -*- coding: utf-8 -*-
"""Earth Observation Scientific Numerical Algorithms.

Pure numerical, vectorized implementations of multispectral indices,
spatial statistics, surface area calculations, and change detection.
Strictly decoupled from deep learning / neural network layers.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np


class EOAlgorithms:
    """Vectorized numerical algorithms for multispectral and raster analysis."""

    @staticmethod
    def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Calculates Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
        nir = np.asarray(nir, dtype=np.float64)
        red = np.asarray(red, dtype=np.float64)
        denominator = nir + red
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = np.where(denominator != 0, (nir - red) / denominator, np.nan)
        return np.clip(ndvi, -1.0, 1.0)

    @staticmethod
    def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculates Normalized Difference Water Index (McFeeters 1996): (Green - NIR) / (Green + NIR)."""
        green = np.asarray(green, dtype=np.float64)
        nir = np.asarray(nir, dtype=np.float64)
        denominator = green + nir
        with np.errstate(divide="ignore", invalid="ignore"):
            ndwi = np.where(denominator != 0, (green - nir) / denominator, np.nan)
        return np.clip(ndwi, -1.0, 1.0)

    @staticmethod
    def compute_nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """Calculates Normalized Burn Ratio: (NIR - SWIR) / (NIR + SWIR)."""
        nir = np.asarray(nir, dtype=np.float64)
        swir = np.asarray(swir, dtype=np.float64)
        denominator = nir + swir
        with np.errstate(divide="ignore", invalid="ignore"):
            nbr = np.where(denominator != 0, (nir - swir) / denominator, np.nan)
        return np.clip(nbr, -1.0, 1.0)

    @staticmethod
    def compute_evi(
        nir: np.ndarray,
        red: np.ndarray,
        blue: np.ndarray,
        g: float = 2.5,
        c1: float = 6.0,
        c2: float = 7.5,
        l: float = 1.0,
    ) -> np.ndarray:
        """Calculates Enhanced Vegetation Index: G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)."""
        nir = np.asarray(nir, dtype=np.float64)
        red = np.asarray(red, dtype=np.float64)
        blue = np.asarray(blue, dtype=np.float64)
        denominator = nir + (c1 * red) - (c2 * blue) + l
        with np.errstate(divide="ignore", invalid="ignore"):
            evi = np.where(denominator != 0, g * (nir - red) / denominator, np.nan)
        return np.clip(evi, -1.0, 1.0)

    @staticmethod
    def compute_savi(nir: np.ndarray, red: np.ndarray, l: float = 0.5) -> np.ndarray:
        """Calculates Soil-Adjusted Vegetation Index: ((NIR - Red) / (NIR + Red + L)) * (1 + L)."""
        nir = np.asarray(nir, dtype=np.float64)
        red = np.asarray(red, dtype=np.float64)
        denominator = nir + red + l
        with np.errstate(divide="ignore", invalid="ignore"):
            savi = np.where(denominator != 0, ((nir - red) / denominator) * (1.0 + l), np.nan)
        return np.clip(savi, -1.0, 1.0)

    @staticmethod
    def compute_spatial_statistics(index_raster: np.ndarray) -> Dict[str, Any]:
        """Extracts robust descriptive spatial statistics from a floating-point index raster."""
        arr = np.asarray(index_raster, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        total = int(arr.size)
        valid_count = int(valid.size)

        if valid_count == 0:
            return {
                "valid_pixel_count": 0,
                "total_pixels": total,
                "valid_pixel_fraction": 0.0,
                "mean": None,
                "std": None,
                "median": None,
                "min": None,
                "max": None,
                "percentiles": {},
            }

        mean_val = float(np.mean(valid))
        std_val = float(np.std(valid))
        median_val = float(np.median(valid))
        min_val = float(np.min(valid))
        max_val = float(np.max(valid))

        return {
            "valid_pixel_count": valid_count,
            "total_pixels": total,
            "valid_pixel_fraction": round(valid_count / total, 4),
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "median": round(median_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "percentiles": {
                "p10": round(float(np.percentile(valid, 10)), 4),
                "p25": round(float(np.percentile(valid, 25)), 4),
                "p75": round(float(np.percentile(valid, 75)), 4),
                "p90": round(float(np.percentile(valid, 90)), 4),
            },
            "total_pixels_evaluated": total,
        }

    @staticmethod
    def calculate_surface_area_km2(
        pixel_count: int,
        pixel_resolution_m: float,
        crs: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculates planar surface area in km² while strictly validating projection.

        Scientifically rejects direct area calculation on angular degree coordinate systems (EPSG:4326).
        """
        if crs:
            crs_lower = str(crs).lower()
            if "epsg:4326" in crs_lower or "degree" in crs_lower or ("wgs 84" in crs_lower and "utm" not in crs_lower):
                return {
                    "error": (
                        "CRS is geographic (degrees). Calculating surface area in square degrees "
                        "is scientifically invalid due to latitude convergence. Reproject to an equal-area or UTM projection."
                    ),
                    "area_km2": None,
                    "crs": crs,
                }

        area_m2 = pixel_count * (pixel_resolution_m ** 2)
        area_km2 = area_m2 / 1_000_000.0
        return {
            "pixel_count": pixel_count,
            "pixel_resolution_m": pixel_resolution_m,
            "area_km2": round(area_km2, 4),
            "crs": crs or "projected_planar",
            "unit": "km²",
        }

    @staticmethod
    def compute_change_detection(
        pre_raster: np.ndarray,
        post_raster: np.ndarray,
        threshold: float = 0.2,
    ) -> Dict[str, Any]:
        """Calculates difference raster (Post - Pre) and significant change mask."""
        pre = np.asarray(pre_raster, dtype=np.float64)
        post = np.asarray(post_raster, dtype=np.float64)
        if pre.shape != post.shape:
            raise ValueError(f"Shape mismatch for change detection: Pre {pre.shape} != Post {post.shape}")

        diff = post - pre
        valid_mask = np.isfinite(diff)
        valid_diff = diff[valid_mask]

        if valid_diff.size == 0:
            return {"error": "No valid overlapping pixels for change detection."}

        loss_mask = (diff < -threshold) & valid_mask
        gain_mask = (diff > threshold) & valid_mask
        unchanged_mask = (~loss_mask) & (~gain_mask) & valid_mask

        total_valid = int(valid_diff.size)
        return {
            "mean_difference": round(float(np.mean(valid_diff)), 4),
            "std_difference": round(float(np.std(valid_diff)), 4),
            "pixels_lost": int(np.sum(loss_mask)),
            "pixels_gained": int(np.sum(gain_mask)),
            "pixels_unchanged": int(np.sum(unchanged_mask)),
            "fraction_lost": round(float(np.sum(loss_mask) / total_valid), 4),
            "fraction_gained": round(float(np.sum(gain_mask) / total_valid), 4),
            "total_pixels_evaluated": total_valid,
        }


# Convenience alias for uniform naming across domain algorithms
EarthObservationAlgorithms = EOAlgorithms
