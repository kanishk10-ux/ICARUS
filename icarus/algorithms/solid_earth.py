# -*- coding: utf-8 -*-
"""Solid Earth and Seismology Scientific Algorithms Layer.

Provides verified, deterministic algorithms for seismic waveform signal processing,
bandpass filtering, signal-to-noise ratio (SNR) calculation, FFT frequency spectrum,
Arias intensity, and Peak Ground Acceleration (PGA) estimation independent of neural networks.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("icarus.algorithms.solid_earth")


class SolidEarthAlgorithms:
    """Core mathematical algorithms for seismology, geophysics, and seismic waveform processing."""

    @staticmethod
    def preprocess_waveform(
        waveform: np.ndarray,
        sampling_rate_hz: float = 100.0,
        taper_fraction: float = 0.05,
    ) -> np.ndarray:
        """Detrends and applies Tukey cosine taper to ends of seismic trace."""
        data = np.asarray(waveform, dtype=np.float64)
        if data.size == 0:
            raise ValueError("Waveform is empty.")

        # Detrend: linear demean
        detrended = data - np.mean(data)

        # Tukey window taper
        n = len(detrended)
        taper_len = int(n * taper_fraction)
        if taper_len > 0:
            window = np.ones(n, dtype=np.float64)
            taper_curve = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, taper_len)))
            window[:taper_len] = taper_curve
            window[-taper_len:] = taper_curve[::-1]
            detrended = detrended * window

        return detrended

    @staticmethod
    def compute_snr_db(
        waveform: np.ndarray,
        noise_window_samples: int = 500,
        signal_window_samples: int = 500,
    ) -> float:
        """Computes Signal-to-Noise Ratio in decibels: 20 * log10(RMS_signal / RMS_noise)."""
        data = np.asarray(waveform, dtype=np.float64)
        if len(data) < noise_window_samples + signal_window_samples:
            raise ValueError(f"Waveform length ({len(data)}) shorter than required noise + signal windows.")

        noise_seg = data[:noise_window_samples]
        signal_seg = data[noise_window_samples : noise_window_samples + signal_window_samples]

        rms_noise = np.sqrt(np.mean(noise_seg ** 2))
        rms_signal = np.sqrt(np.mean(signal_seg ** 2))

        if rms_noise == 0 or np.isnan(rms_noise):
            return 100.0  # Infinite SNR proxy

        snr_ratio = max(1e-6, rms_signal / rms_noise)
        snr_db = 20.0 * np.log10(snr_ratio)
        return round(float(snr_db), 2)

    @staticmethod
    def compute_frequency_spectrum(
        waveform: np.ndarray,
        sampling_rate_hz: float = 100.0,
    ) -> Dict[str, Any]:
        """Calculates FFT amplitude spectrum and identifies dominant frequency (Hz)."""
        data = np.asarray(waveform, dtype=np.float64)
        n = len(data)
        if n < 8:
            return {"error": "Waveform too short for spectral decomposition."}

        # Demean before FFT
        demeaned = data - np.mean(data)
        fft_vals = np.fft.rfft(demeaned)
        freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
        amplitudes = np.abs(fft_vals) / n

        # Find dominant frequency
        dom_idx = int(np.argmax(amplitudes[1:])) + 1  # Ignore DC component
        dom_freq = float(freqs[dom_idx])
        peak_amp = float(amplitudes[dom_idx])

        # Mean frequency
        total_amp = np.sum(amplitudes)
        mean_freq = float(np.sum(freqs * amplitudes) / total_amp) if total_amp > 0 else 0.0

        return {
            "dominant_frequency_hz": round(dom_freq, 3),
            "mean_frequency_hz": round(mean_freq, 3),
            "peak_spectral_amplitude": round(peak_amp, 6),
            "nyquist_frequency_hz": round(sampling_rate_hz / 2.0, 1),
            "total_frequency_bins": len(freqs),
            "unit_frequency": "Hz",
        }

    @staticmethod
    def compute_ground_motion_metrics(
        acceleration_trace_m_per_s2: np.ndarray,
        sampling_rate_hz: float = 100.0,
    ) -> Dict[str, Any]:
        """Calculates Peak Ground Acceleration (PGA) in g, PGV, and Arias Intensity (m/s).

        Arias Intensity Ia = (pi / (2 * g)) * integral(a(t)^2 dt)
        """
        acc = np.asarray(acceleration_trace_m_per_s2, dtype=np.float64)
        if acc.size == 0:
            return {"error": "Acceleration trace is empty."}

        dt = 1.0 / sampling_rate_hz
        g = 9.80665  # Standard gravity m/s²

        # Peak Ground Acceleration
        pga_mps2 = float(np.max(np.abs(acc)))
        pga_g = pga_mps2 / g

        # Numerical integration for velocity: v(t) = cumtrapz(a(t), dt)
        vel = np.cumsum(acc) * dt
        vel_detrend = vel - np.mean(vel)
        pgv_mps = float(np.max(np.abs(vel_detrend)))

        # Arias Intensity
        arias_intensity = (np.pi / (2.0 * g)) * np.sum(acc ** 2) * dt

        return {
            "pga_mps2": round(pga_mps2, 4),
            "pga_g": round(pga_g, 5),
            "pgv_mps": round(pgv_mps, 4),
            "arias_intensity_mps": round(float(arias_intensity), 5),
            "duration_seconds": round(len(acc) * dt, 2),
            "sampling_rate_hz": sampling_rate_hz,
            "unit_pga": "g",
            "unit_pgv": "m/s",
            "unit_arias": "m/s",
        }
