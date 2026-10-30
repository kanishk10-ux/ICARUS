# -*- coding: utf-8 -*-
"""Astronomy and Astrophysics Scientific Algorithms Layer.

Provides verified, deterministic algorithms for stellar light curve statistics,
Lomb-Scargle periodicity detection, aperture photometry, signal-to-noise estimation,
and galaxy CAS (Concentration-Asymmetry) structural analysis independent of neural networks.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("icarus.algorithms.astronomy")


class AstronomyAlgorithms:
    """Core mathematical algorithms for observational astronomy, astrophysics, and photometry."""

    @staticmethod
    def analyze_light_curve(
        times_days: np.ndarray,
        flux: np.ndarray,
        min_period_days: float = 0.1,
        max_period_days: float = 50.0,
        num_frequency_evals: int = 200,
    ) -> Dict[str, Any]:
        """Analyzes time-series stellar photometry and computes Lomb-Scargle periodogram peak."""
        t = np.asarray(times_days, dtype=np.float64)
        f = np.asarray(flux, dtype=np.float64)

        valid = np.isfinite(t) & np.isfinite(f)
        t_clean, f_clean = t[valid], f[valid]

        if t_clean.size < 5:
            return {"error": "Light curve requires at least 5 valid observations."}

        mean_flux = float(np.mean(f_clean))
        std_flux = float(np.std(f_clean))
        amplitude = float(np.max(f_clean) - np.min(f_clean))

        # Relative variability amplitude
        variability_amp = amplitude / mean_flux if mean_flux > 0 else 0.0

        # Lomb-Scargle Periodogram Implementation (Scargle 1982)
        f_norm = f_clean - mean_flux
        freqs = np.linspace(1.0 / max_period_days, 1.0 / min_period_days, num_frequency_evals)
        powers = []

        for freq in freqs:
            omega = 2.0 * np.pi * freq
            # Phase tau
            tau = np.arctan2(np.sum(np.sin(2.0 * omega * t_clean)), np.sum(np.cos(2.0 * omega * t_clean))) / (2.0 * omega)
            c = np.cos(omega * (t_clean - tau))
            s = np.sin(omega * (t_clean - tau))

            sum_c2 = np.sum(c ** 2)
            sum_s2 = np.sum(s ** 2)

            p_c = (np.sum(f_norm * c) ** 2) / sum_c2 if sum_c2 > 0 else 0.0
            p_s = (np.sum(f_norm * s) ** 2) / sum_s2 if sum_s2 > 0 else 0.0
            power = 0.5 * (p_c + p_s) / (std_flux ** 2) if std_flux > 0 else 0.0
            powers.append(power)

        powers = np.array(powers)
        best_idx = int(np.argmax(powers))
        best_freq = float(freqs[best_idx])
        best_period = float(1.0 / best_freq)
        peak_power = float(powers[best_idx])

        return {
            "mean_flux": round(mean_flux, 4),
            "std_flux": round(std_flux, 4),
            "flux_amplitude": round(amplitude, 4),
            "relative_variability": round(float(variability_amp), 4),
            "detected_period_days": round(best_period, 4),
            "dominant_frequency_cycles_per_day": round(best_freq, 4),
            "periodogram_peak_power": round(peak_power, 4),
            "is_periodic": peak_power > 10.0,
            "total_observations": int(t_clean.size),
            "time_span_days": round(float(np.max(t_clean) - np.min(t_clean)), 2),
        }

    @staticmethod
    def compute_aperture_photometry(
        image: np.ndarray,
        center_x: int,
        center_y: int,
        aperture_radius_pixels: float = 10.0,
        background_inner_radius: float = 15.0,
        background_outer_radius: float = 25.0,
        zero_point_magnitude: float = 25.0,
    ) -> Dict[str, Any]:
        """Performs circular aperture photometry with local sky background subtraction."""
        img = np.asarray(image, dtype=np.float64)
        if img.ndim != 2:
            return {"error": "Astronomical image must be a 2D intensity array."}

        h, w = img.shape
        y, x = np.ogrid[:h, :w]
        dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2

        # Masks
        aperture_mask = dist_sq <= (aperture_radius_pixels ** 2)
        sky_mask = (dist_sq >= (background_inner_radius ** 2)) & (dist_sq <= (background_outer_radius ** 2))

        # Local sky background per pixel
        sky_pixels = img[sky_mask]
        if sky_pixels.size == 0:
            return {"error": "Sky background annulus falls entirely outside image bounds."}

        sky_median = float(np.median(sky_pixels))
        sky_std = float(np.std(sky_pixels))

        # Sum inside aperture
        aperture_pixels = img[aperture_mask]
        n_aperture = int(aperture_pixels.size)
        raw_flux = float(np.sum(aperture_pixels))

        # Background subtracted net flux
        net_flux = raw_flux - (n_aperture * sky_median)

        # Photometric SNR
        # Poisson noise approximation: SNR = net_flux / sqrt(net_flux + n_aperture * sky_std^2)
        variance = max(1.0, net_flux + n_aperture * (sky_std ** 2))
        snr = net_flux / np.sqrt(variance) if net_flux > 0 else 0.0

        # Instrumental apparent magnitude
        magnitude = None
        if net_flux > 0:
            magnitude = zero_point_magnitude - 2.5 * np.log10(net_flux)

        return {
            "net_source_flux": round(net_flux, 2),
            "sky_background_per_pixel": round(sky_median, 4),
            "sky_background_std": round(sky_std, 4),
            "photometric_snr": round(float(snr), 2),
            "instrumental_magnitude": round(float(magnitude), 3) if magnitude is not None else None,
            "zero_point_applied": zero_point_magnitude,
            "aperture_pixel_count": n_aperture,
        }

    @staticmethod
    def compute_galaxy_cas_asymmetry(galaxy_image: np.ndarray) -> Dict[str, Any]:
        """Calculates Conselice (2003) structural asymmetry parameter A from 180-degree rotation.

        A = sum(|I_0 - I_180|) / (2 * sum(|I_0|))
        Normal galaxies: A < 0.35. Disturbed / merging galaxies: A >= 0.35.
        """
        img = np.asarray(galaxy_image, dtype=np.float64)
        if img.ndim != 2:
            return {"error": "Galaxy image must be a 2D intensity cutout."}

        # Demean by background level
        bg_level = float(np.percentile(img, 10.0))
        subtracted = np.maximum(0.0, img - bg_level)

        # Rotate 180 degrees
        rotated_180 = np.rot90(subtracted, 2)

        # Difference
        diff = np.abs(subtracted - rotated_180)
        total_intensity = np.sum(subtracted)

        if total_intensity <= 0:
            return {"error": "Galaxy signal has zero flux after background subtraction."}

        asymmetry = float(np.sum(diff) / (2.0 * total_intensity))

        return {
            "asymmetry_index_a": round(asymmetry, 4),
            "is_disturbed_or_merger": asymmetry >= 0.35,
            "morphology_indicator": "disturbed_merger" if asymmetry >= 0.35 else "regular_symmetric",
            "total_galaxy_flux": round(float(total_intensity), 2),
        }
