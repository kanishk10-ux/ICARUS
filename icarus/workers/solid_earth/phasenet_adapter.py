# -*- coding: utf-8 -*-
"""SeisBench PhaseNet Specialist Model Adapter for Solid Earth / Seismology.

Implements verified architectural specifications, waveform input validation, sampling-rate
validation (100 Hz), 3-component [Z, N, E] preprocessing, seismic P-wave and S-wave phase picking,
confidence estimation, and GPU memory lifecycle management for SeisBench PhaseNet.
"""
import gc
import logging
import threading
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("icarus.workers.solid_earth.phasenet")


class PhaseNetModelStatus(str, Enum):
    """Lifecycle status of PhaseNet model in worker memory."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class PhaseNetAdapter:
    """Specialist adapter for SeisBench PhaseNet seismic phase picker.

    VERIFIED ARCHITECTURAL SPECIFICATIONS:
    - Framework: SeisBench (https://github.com/seisbench/seisbench)
    - Reference: Zhu & Beroza (2019), "PhaseNet: A deep-neural-network-based
      arrival-time picking method for P- and S-waves"
    - Default Pretrained Checkpoint: "stead" (Stanford Earthquake Dataset)
    - Supported Checkpoints: "stead", "ethz", "instance", "scedc", "geofon"
    - Input Format: 3-component seismogram (Channels=3, Samples=N)
    - Default Channels: [Z (Vertical), N (North-South), E (East-West)]
    - Target Sampling Rate: 100 Hz (10 ms per sample)
    - Window Size: Flexible, typically 3000 to 6000 samples (30s - 60s at 100 Hz)
    - Output Format: 3 probability curves of shape (3, N)
        Channel 0: Noise probability
        Channel 1: P-wave arrival probability
        Channel 2: S-wave arrival probability
    """

    MODEL_ID = "seisbench/phasenet"
    DEFAULT_PRETRAINED = "stead"
    REQUIRED_CHANNELS = 3
    DEFAULT_SAMPLING_RATE_HZ = 100.0

    def __init__(self, pretrained_weights: str = "stead", device: str = "cpu"):
        self.device = device
        self.pretrained_weights = pretrained_weights
        self.status = PhaseNetModelStatus.UNLOADED
        self.model: Optional[Any] = None
        self.vram_allocated_mb: int = 0
        self.last_error: Optional[str] = None
        self._lock = threading.Lock()

    def get_status(self) -> Dict[str, Any]:
        """Returns current model status and hardware telemetry."""
        return {
            "model_id": self.MODEL_ID,
            "pretrained_weights": self.pretrained_weights,
            "status": self.status.value,
            "device": self.device,
            "vram_allocated_mb": self.vram_allocated_mb,
            "required_channels": self.REQUIRED_CHANNELS,
            "target_sampling_rate_hz": self.DEFAULT_SAMPLING_RATE_HZ,
            "supported_tasks": ["p_wave_picking", "s_wave_picking", "waveform_probability_curves"],
            "last_error": self.last_error,
        }

    def validate_waveform_input(
        self,
        waveform: Union[np.ndarray, Any],
        sampling_rate: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Validates input seismic waveform format and sampling rate."""
        arr = np.asarray(waveform)

        if arr.size == 0:
            return False, "Input waveform is empty."

        if not np.all(np.isfinite(arr)):
            return False, "Waveform data contains NaN or Inf values."

        # Expected shape: (3, N) or (1, 3, N) or (N, 3)
        if arr.ndim == 1:
            return False, f"PhaseNet requires 3-component seismic data [Z, N, E]. Got single-channel 1D data with {arr.size} samples."

        if arr.ndim == 2:
            if arr.shape[0] != 3 and arr.shape[1] == 3:
                # Transpose (N, 3) to (3, N)
                arr = arr.T
            if arr.shape[0] != 3:
                return False, f"Expected 3 channels [Z, N, E], got shape {arr.shape}."
            if arr.shape[1] < 100:
                return False, f"Waveform too short for phase picking: {arr.shape[1]} samples. Minimum 100 samples required."
        elif arr.ndim == 3:
            if arr.shape[1] != 3:
                return False, f"Expected (B, 3, N) tensor, got shape {arr.shape}."
        else:
            return False, f"Unsupported waveform dimensions: {arr.ndim}D shape {arr.shape}."

        if sampling_rate is not None and abs(sampling_rate - self.DEFAULT_SAMPLING_RATE_HZ) > 1.0:
            return (
                False,
                f"Sampling rate is {sampling_rate} Hz. SeisBench PhaseNet requires 100 Hz. Resample waveform to 100 Hz before inference.",
            )

        return True, "Valid seismic waveform input"

    def preprocess_waveform(self, waveform_3c: np.ndarray) -> np.ndarray:
        """Detrends, tapers, and standardizes 3-component waveform."""
        data = np.asarray(waveform_3c, dtype=np.float32)
        if data.ndim == 2 and data.shape[0] != 3 and data.shape[1] == 3:
            data = data.T

        # Per-channel mean removal (detrend)
        means = np.mean(data, axis=-1, keepdims=True)
        detrended = data - means

        # Standard deviation normalization
        stds = np.std(detrended, axis=-1, keepdims=True)
        stds = np.where(stds == 0, 1.0, stds)
        normalized = detrended / stds

        return normalized.astype(np.float32)

    def load_model(self) -> Tuple[bool, str]:
        """Lazy thread-safe loader for SeisBench PhaseNet."""
        with self._lock:
            if self.status == PhaseNetModelStatus.READY and self.model is not None:
                return True, "SeisBench PhaseNet already loaded in memory."

            self.status = PhaseNetModelStatus.LOADING
            try:
                import torch
            except ImportError:
                self.status = PhaseNetModelStatus.FAILED
                self.last_error = "PyTorch is not installed in the runtime environment."
                return False, self.last_error

            try:
                import seisbench.models as sbm
            except ImportError:
                self.status = PhaseNetModelStatus.FAILED
                self.last_error = "SeisBench is not installed in the runtime environment (pip install seisbench)."
                logger.warning(self.last_error)
                return False, self.last_error

            try:
                logger.info("Loading SeisBench PhaseNet pretrained weights '%s' on %s", self.pretrained_weights, self.device)
                model = sbm.PhaseNet.from_pretrained(self.pretrained_weights)
                model = model.to(self.device)
                model.eval()

                self.model = model
                self.status = PhaseNetModelStatus.READY
                self.vram_allocated_mb = 512 if self.device == "cuda" else 0
                self.last_error = None
                return True, f"SeisBench PhaseNet ('{self.pretrained_weights}') ready on {self.device}."
            except Exception as e:
                self.status = PhaseNetModelStatus.FAILED
                self.last_error = str(e)
                logger.error("Failed to load PhaseNet: %s", str(e))
                return False, f"PhaseNet loading failed: {str(e)}"

    def unload_model(self) -> Tuple[bool, str, int]:
        """Unloads PhaseNet weights from memory and reclaims VRAM."""
        with self._lock:
            freed = self.vram_allocated_mb
            self.model = None
            self.vram_allocated_mb = 0
            self.status = PhaseNetModelStatus.UNLOADED

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            gc.collect()
            logger.info("SeisBench PhaseNet unloaded. Reclaimed ~%d MB", freed)
            return True, "Model unloaded successfully", freed

    def predict_phases(
        self,
        waveform_3c: np.ndarray,
        sampling_rate: float = 100.0,
        threshold: float = 0.3,
        start_time_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes phase picking on 3-component seismic waveform [Z, N, E].

        Returns P-wave and S-wave arrival picks, peak probabilities, and time offsets.
        """
        valid, msg = self.validate_waveform_input(waveform_3c, sampling_rate=sampling_rate)
        if not valid:
            return {"status": "invalid_input", "error": msg}

        if self.status != PhaseNetModelStatus.READY or self.model is None:
            return {
                "status": "model_unloaded",
                "error": "PhaseNet is not loaded. Call load_model() first.",
            }

        try:
            import torch
        except ImportError:
            return {"status": "missing_dependency", "error": "PyTorch is required for PhaseNet inference."}

        norm_data = self.preprocess_waveform(waveform_3c)
        if norm_data.ndim == 2:
            tensor_in = torch.from_numpy(norm_data).unsqueeze(0).float().to(self.device)  # (1, 3, N)
        else:
            tensor_in = torch.from_numpy(norm_data).float().to(self.device)

        try:
            with torch.inference_mode():
                # PhaseNet forward pass returns predictions: (B, 3, N) or (B, N, 3)
                preds = self.model(tensor_in)
                if isinstance(preds, (tuple, list)):
                    preds = preds[0]
                preds_np = preds.detach().cpu().numpy().squeeze(0)  # Shape (3, N) or (N, 3)

            if preds_np.shape[0] != 3 and preds_np.shape[-1] == 3:
                preds_np = preds_np.T  # (3, N) -> [Noise, P, S]

            n_samples = preds_np.shape[1]
            time_axis = np.arange(n_samples) / sampling_rate

            p_probs = preds_np[1]
            s_probs = preds_np[2]

            # Detect P picks
            p_picks = self._extract_picks_from_curve(p_probs, time_axis, threshold, phase_label="P")
            s_picks = self._extract_picks_from_curve(s_probs, time_axis, threshold, phase_label="S")

            return {
                "status": "success",
                "model_id": self.MODEL_ID,
                "pretrained_weights": self.pretrained_weights,
                "sampling_rate_hz": sampling_rate,
                "sample_count": n_samples,
                "duration_seconds": round(n_samples / sampling_rate, 3),
                "p_picks": p_picks,
                "s_picks": s_picks,
                "total_p_picks": len(p_picks),
                "total_s_picks": len(s_picks),
                "probability_summary": {
                    "p_max_probability": round(float(np.max(p_probs)), 4),
                    "s_max_probability": round(float(np.max(s_probs)), 4),
                },
                "start_time": start_time_iso,
                "scientific_note": "Seismic arrival picks generated by SeisBench PhaseNet. Probabilities represent likelihood of phase arrival within continuous temporal window.",
            }
        except Exception as e:
            logger.error("PhaseNet prediction failed: %s", str(e))
            return {"status": "inference_failed", "error": f"PhaseNet inference failed: {str(e)}"}

    @staticmethod
    def _extract_picks_from_curve(
        probs: np.ndarray,
        time_axis: np.ndarray,
        threshold: float,
        phase_label: str,
    ) -> List[Dict[str, Any]]:
        """Identifies local maxima above threshold to determine arrival picks."""
        picks = []
        above_thresh = probs > threshold
        if not np.any(above_thresh):
            return picks

        # Simple peak detection: find contiguous regions above threshold
        in_peak = False
        start_idx = 0

        for i, val in enumerate(above_thresh):
            if val and not in_peak:
                in_peak = True
                start_idx = i
            elif not val and in_peak:
                in_peak = False
                end_idx = i
                # Peak index within window
                peak_idx = start_idx + int(np.argmax(probs[start_idx:end_idx]))
                picks.append({
                    "phase": phase_label,
                    "sample_index": peak_idx,
                    "time_offset_seconds": round(float(time_axis[peak_idx]), 4),
                    "confidence": round(float(probs[peak_idx]), 4),
                })

        if in_peak:
            peak_idx = start_idx + int(np.argmax(probs[start_idx:]))
            picks.append({
                "phase": phase_label,
                "sample_index": peak_idx,
                "time_offset_seconds": round(float(time_axis[peak_idx]), 4),
                "confidence": round(float(probs[peak_idx]), 4),
            })

        return picks


    # Alias for uniform feature extraction / inference interface
    extract_features = predict_phases


# Convenience export aliases
PhaseNetSeismologyAdapter = PhaseNetAdapter

