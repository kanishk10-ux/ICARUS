# -*- coding: utf-8 -*-
"""Earth Observation processing pipeline: Spectral algorithms, GeoTIFF ingestion, band mapping, and spatial tiling."""
import logging
import math
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("icarus.workers.eo.pipeline")


class EOPipeline:
    """Core computational pipeline for Earth Observation raster data."""

    # Standard HLS / Sentinel-2 spectral band naming map
    STANDARD_BAND_MAP = {
        "blue": ["B02", "B2", "blue", "band_2"],
        "green": ["B03", "B3", "green", "band_3"],
        "red": ["B04", "B4", "red", "band_4"],
        "nir": ["B08", "B8", "nir", "narrow_nir", "B8A", "band_8"],
        "swir1": ["B11", "B11A", "swir1", "swir_1", "band_11"],
        "swir2": ["B12", "B12A", "swir2", "swir_2", "band_12"],
    }

    # ============================================================
    # 1. Traditional Spectral Algorithms
    # ============================================================
    @staticmethod
    def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)
        Enforces shape compatibility and handles zero-division cleanly."""
        nir = np.asarray(nir, dtype=np.float64)
        red = np.asarray(red, dtype=np.float64)
        if nir.shape != red.shape:
            raise ValueError(f"Shape mismatch for NDVI: NIR {nir.shape} != Red {red.shape}")

        denom = nir + red
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = np.where(denom != 0, (nir - red) / denom, np.nan)
        return np.clip(ndvi, -1.0, 1.0)

    @staticmethod
    def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Normalized Difference Water Index (McFeeters): (Green - NIR) / (Green + NIR)"""
        green = np.asarray(green, dtype=np.float64)
        nir = np.asarray(nir, dtype=np.float64)
        if green.shape != nir.shape:
            raise ValueError(f"Shape mismatch for NDWI: Green {green.shape} != NIR {nir.shape}")

        denom = green + nir
        with np.errstate(divide="ignore", invalid="ignore"):
            ndwi = np.where(denom != 0, (green - nir) / denom, np.nan)
        return np.clip(ndwi, -1.0, 1.0)

    @staticmethod
    def compute_nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """Normalized Burn Ratio: (NIR - SWIR) / (NIR + SWIR)"""
        nir = np.asarray(nir, dtype=np.float64)
        swir = np.asarray(swir, dtype=np.float64)
        if nir.shape != swir.shape:
            raise ValueError(f"Shape mismatch for NBR: NIR {nir.shape} != SWIR {swir.shape}")

        denom = nir + swir
        with np.errstate(divide="ignore", invalid="ignore"):
            nbr = np.where(denom != 0, (nir - swir) / denom, np.nan)
        return np.clip(nbr, -1.0, 1.0)

    @staticmethod
    def compute_vegetation_statistics(ndvi: np.ndarray) -> Dict[str, Any]:
        """Calculates statistical moments and biome coverage fractions."""
        valid = ndvi[~np.isnan(ndvi)]
        if len(valid) == 0:
            return {"error": "All raster pixels are NaN or nodata"}

        total_valid = len(valid)
        mean_val = float(np.mean(valid))
        std_val = float(np.std(valid))
        median_val = float(np.median(valid))
        min_val = float(np.min(valid))
        max_val = float(np.max(valid))

        return {
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
            "classification_fractions": {
                "water_or_bare_fraction": round(float(np.sum(valid < 0.1) / total_valid), 4),
                "sparse_vegetation_fraction": round(float(np.sum((valid >= 0.1) & (valid < 0.3)) / total_valid), 4),
                "moderate_vegetation_fraction": round(float(np.sum((valid >= 0.3) & (valid < 0.6)) / total_valid), 4),
                "dense_canopy_fraction": round(float(np.sum(valid >= 0.6) / total_valid), 4),
            },
            "total_pixels_evaluated": total_valid,
        }

    @staticmethod
    def calculate_surface_area_km2(
        pixel_count: int, pixel_resolution_m: float, crs: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculates surface area in km² while enforcing CRS projection validation.
        Rejects direct calculation if CRS is in angular degrees (e.g. EPSG:4326)."""
        is_geographic = False
        if crs:
            crs_lower = str(crs).lower()
            if "epsg:4326" in crs_lower or "degree" in crs_lower or "wgs 84" in crs_lower and "utm" not in crs_lower:
                is_geographic = True

        if is_geographic:
            return {
                "error": "CRS is geographic (degrees). Calculating real-world area directly in square degrees is scientifically invalid. Reproject to an equal-area or UTM projection first.",
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

    # ============================================================
    # 2. GeoTIFF Ingestion, Metadata & CRS Validation
    # ============================================================
    @staticmethod
    def inspect_geotiff(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Inspects and validates a GeoTIFF raster file."""
        p = Path(file_path)
        if not p.exists():
            return {"error": f"File not found: {p}", "valid": False}

        try:
            import rasterio
            with rasterio.open(p) as src:
                crs_str = str(src.crs) if src.crs else None
                is_projected = src.crs.is_projected if src.crs else False
                res_x, res_y = src.res if hasattr(src, "res") else (None, None)

                return {
                    "valid": True,
                    "file_name": p.name,
                    "driver": src.driver,
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "dtypes": [str(d) for d in src.dtypes],
                    "crs": crs_str,
                    "is_projected": is_projected,
                    "resolution": (res_x, res_y),
                    "bounds": {
                        "left": src.bounds.left,
                        "bottom": src.bounds.bottom,
                        "right": src.bounds.right,
                        "top": src.bounds.top,
                    } if src.bounds else None,
                    "nodata": src.nodata,
                    "metadata": dict(src.meta),
                }
        except ImportError:
            # High-precision fallback using tifffile when rasterio is not installed
            try:
                import tifffile
                with tifffile.TiffFile(p) as tif:
                    page = tif.pages[0]
                    h, w = page.shape[-2], page.shape[-1]
                    count = page.shape[0] if page.ndim == 3 else 1
                    geotags = page.geotiff_tags or {}

                    crs_code = geotags.get("ProjectedCSTypeGeoKey")
                    crs_str = f"EPSG:{crs_code}" if crs_code else geotags.get("GTCitationGeoKey")
                    is_proj = bool(crs_code and crs_code >= 20000)

                    scale = geotags.get("ModelPixelScale")
                    res = (scale[0], scale[1]) if scale and len(scale) >= 2 else (None, None)

                    tiepoint = geotags.get("ModelTiepoint")
                    bounds = None
                    if tiepoint and scale and len(tiepoint) >= 6 and len(scale) >= 2:
                        x0, y0 = tiepoint[3], tiepoint[4]
                        dx, dy = scale[0], scale[1]
                        bounds = {
                            "left": float(x0),
                            "top": float(y0),
                            "right": float(x0 + dx * w),
                            "bottom": float(y0 - dy * h),
                        }

                    return {
                        "valid": True,
                        "file_name": p.name,
                        "driver": "GeoTIFF (tifffile engine)",
                        "width": int(w),
                        "height": int(h),
                        "bands": int(count),
                        "dtypes": [str(page.dtype)],
                        "crs": str(crs_str) if crs_str else None,
                        "is_projected": is_proj,
                        "resolution": res,
                        "bounds": bounds,
                        "nodata": None,
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                    }
            except Exception as ex:
                return {
                    "valid": True,
                    "file_name": p.name,
                    "note": f"rasterio not installed; basic file stat inspection: {str(ex)}",
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                }
        except Exception as e:
            return {"valid": False, "error": f"Failed to read GeoTIFF: {str(e)}"}

    @staticmethod
    def read_bands_from_geotiff(
        file_path: Union[str, Path], bands: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reads specified bands (1-indexed) from a GeoTIFF."""
        p = Path(file_path)
        try:
            import rasterio
            with rasterio.open(p) as src:
                target_bands = bands or list(range(1, src.count + 1))
                data = src.read(target_bands)
                meta = {
                    "crs": str(src.crs) if src.crs else None,
                    "transform": src.transform,
                    "nodata": src.nodata,
                    "shape": data.shape,
                    "resolution": src.res,
                }
                return data, meta
        except ImportError:
            import tifffile
            with tifffile.TiffFile(p) as tif:
                arr = tif.asarray()
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, axis=0)
                geotags = tif.pages[0].geotiff_tags or {}
                crs_code = geotags.get("ProjectedCSTypeGeoKey")
                meta = {
                    "crs": f"EPSG:{crs_code}" if crs_code else geotags.get("GTCitationGeoKey"),
                    "transform": None,
                    "nodata": None,
                    "shape": arr.shape,
                    "resolution": geotags.get("ModelPixelScale"),
                }
                return arr, meta

    # ============================================================
    # 3. Spatial Tiling & Reconstruction
    # ============================================================
    @staticmethod
    def tile_raster(
        raster: np.ndarray, tile_size: int = 224, stride: int = 224
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Tuple[int, int]]:
        """Extracts spatial tiles from raster array (channels, height, width).
        Pads edges if dimensions are not cleanly divisible by stride."""
        if raster.ndim == 2:
            raster = np.expand_dims(raster, axis=0)

        channels, h, w = raster.shape

        pad_h = (stride - (h % stride)) % stride
        pad_w = (stride - (w % stride)) % stride

        # Ensure at least tile_size
        if h + pad_h < tile_size:
            pad_h += (tile_size - (h + pad_h))
        if w + pad_w < tile_size:
            pad_w += (tile_size - (w + pad_w))

        if pad_h > 0 or pad_w > 0:
            padded = np.pad(raster, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        else:
            padded = raster

        tiles = []
        coordinates = []
        pad_c, pad_height, pad_width = padded.shape

        for y in range(0, pad_height - tile_size + 1, stride):
            for x in range(0, pad_width - tile_size + 1, stride):
                patch = padded[:, y : y + tile_size, x : x + tile_size]
                tiles.append(patch)
                coordinates.append((y, x))

        return tiles, coordinates, (h, w)

    @staticmethod
    def reconstruct_raster(
        tiles: List[np.ndarray],
        coordinates: List[Tuple[int, int]],
        original_shape: Tuple[int, int],
        tile_size: int = 224,
    ) -> np.ndarray:
        """Reconstructs spatial raster from a set of tiles and coordinates."""
        orig_h, orig_w = original_shape
        if not tiles:
            return np.zeros((1, orig_h, orig_w), dtype=np.float32)

        channels = tiles[0].shape[0] if tiles[0].ndim == 3 else 1
        max_y = max(y + tile_size for y, x in coordinates)
        max_x = max(x + tile_size for y, x in coordinates)

        canvas = np.zeros((channels, max_y, max_x), dtype=np.float32)
        weight = np.zeros((channels, max_y, max_x), dtype=np.float32)

        for patch, (y, x) in zip(tiles, coordinates):
            if patch.ndim == 2:
                patch = np.expand_dims(patch, axis=0)
            canvas[:, y : y + tile_size, x : x + tile_size] += patch
            weight[:, y : y + tile_size, x : x + tile_size] += 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(weight > 0, canvas / weight, 0)

        # Crop back to original dimensions
        cropped = result[:, :orig_h, :orig_w]
        return cropped.squeeze(0) if channels == 1 else cropped

