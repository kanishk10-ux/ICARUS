# -*- coding: utf-8 -*-
"""IBM/NASA Prithvi-EO-2.0-300M Foundation Model Adapter.

Implements verified architectural specifications, official HuggingFace checkpoint loading,
dynamic positional embedding pruning, strict 6-band / 4-frame shape validation,
climatological reflectance normalization, and GPU memory lifecycle management for
Prithvi-EO-2.0-300M (IBM/NASA Geospatial Foundation Model).
"""
import gc
import logging
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("icarus.workers.eo.prithvi")


class PrithviModelStatus(str, Enum):
    """Lifecycle status of Prithvi foundation model in worker memory."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class PrithviEOAdapter:
    """Specialist adapter for IBM/NASA Prithvi-EO-2.0-300M foundation model.

    VERIFIED ARCHITECTURAL SPECIFICATIONS:
    - Source: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M
    - Checkpoint: Prithvi_EO_V2_300M.pt
    - HF Revision: 9eb1b1102806593963daa333bcc491b1c6f8562f
    - Architecture: PrithviMAE / 3D Temporal Vision Transformer (ViT)
    - Total Parameter Count: 330,419,712
    - Patch Size: (1, 16, 16) [Temporal patch=1, Spatial patch=16x16]
    - Spatial Dimensions: Exactly 224 x 224 pixels
    - Temporal Dimension: Exactly 4 timesteps (T=4)
    - Channels: Exactly 6 spectral bands in fixed order:
        Channel 0: Blue (B02)
        Channel 1: Green (B03)
        Channel 2: Red (B04)
        Channel 3: Vegetation Red Edge / Narrow NIR (B05)
        Channel 4: SWIR 1 (B06)
        Channel 5: SWIR 2 (B07)
    - Input Tensor Shape: (Batch, Channels=6, Time_Frames=4, Height=224, Width=224)
    - Output Token Dimensions:
        * Full encoder tokens: (Batch, 785, 1024) [1 CLS token + 784 patch tokens]
        * Patch embeddings only: (Batch, 784, 1024) [4 frames * 14 * 14 spatial patches = 784]
    - Pretrained Base Output: Latent representation tokens.
      (NOTE: The base model is an encoder backbone; it does NOT produce segmentation
       masks or classifications without a trained downstream task head).
    """

    MODEL_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
    LEGACY_MODEL_ID = "prithvi-eo-2.0-300m"
    HF_REPO_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
    CHECKPOINT_FILENAME = "Prithvi_EO_V2_300M.pt"
    HF_REVISION = "9eb1b1102806593963daa333bcc491b1c6f8562f"
    PARAMETER_COUNT = 330419712

    # Verified Official Model Configuration
    OFFICIAL_CONFIG: Dict[str, Any] = {
        "img_size": 224,
        "num_frames": 4,
        "patch_size": [1, 16, 16],
        "in_chans": 6,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "mlp_ratio": 4,
        "coords_encoding": [],
        "coords_scale_learn": False,
        "mask_ratio": 0.75,
        "norm_pix_loss": False,
        "bands": ["B02", "B03", "B04", "B05", "B06", "B07"],
        "mean": [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
        "std": [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
    }

    REQUIRED_BANDS: List[str] = OFFICIAL_CONFIG["bands"]
    PATCH_SIZE: int = 16
    IMAGE_SIZE: int = 224
    NUM_FRAMES: int = 4
    IN_CHANNELS: int = 6
    EMBED_DIM: int = 1024

    # Official Normalization Constants
    CHANNEL_MEANS = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32)
    CHANNEL_STDS = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32)

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.status = PrithviModelStatus.UNLOADED
        self.model: Optional[Any] = None
        self.vram_allocated_mb: int = 0
        self.last_error: Optional[str] = None
        self._lock = threading.Lock()

    def get_status(self) -> Dict[str, Any]:
        """Returns current model loading, configuration, and hardware status."""
        return {
            "model_id": self.MODEL_ID,
            "status": self.status.value,
            "device": self.device,
            "parameter_count": self.PARAMETER_COUNT if self.status == PrithviModelStatus.READY else 0,
            "vram_allocated_mb": self.vram_allocated_mb,
            "bands_required": self.REQUIRED_BANDS,
            "num_frames_required": self.NUM_FRAMES,
            "input_shape_required": [1, self.IN_CHANNELS, self.NUM_FRAMES, self.IMAGE_SIZE, self.IMAGE_SIZE],
            "supported_tasks": ["feature_extraction", "latent_representation"],
            "downstream_heads_available": {
                "flood_mapping": False,  # Requires Sen1Floods11 task head
                "burn_scars": False,     # Requires BurnScars task head
                "land_cover": False,     # Requires multi-class segmentation task head
            },
            "last_error": self.last_error,
        }

    def validate_input_tensor(self, tensor: Union[np.ndarray, Any]) -> Tuple[bool, str]:
        """Validates input tensor conformity with official Prithvi-EO-2.0-300M requirements.
        
        Strictly checks:
        1. Non-empty and finite values (no NaN or Inf).
        2. Tensor dimensions: (B, C, T, H, W) where C=6, T=4, H=224, W=224.
        """
        # Convert torch tensor to numpy for inspection if necessary
        is_torch = False
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                is_torch = True
                if not bool(torch.all(torch.isfinite(tensor)).item()):
                    return False, "Input tensor contains NaN or Inf values."
                if tensor.numel() == 0:
                    return False, "Input tensor is empty."
                shape = list(tensor.shape)
        except ImportError:
            pass

        if not is_torch:
            arr = np.asarray(tensor)
            if arr.size == 0:
                return False, "Input tensor is empty."
            if not np.all(np.isfinite(arr)):
                return False, "Input tensor contains NaN or Inf values."
            shape = list(arr.shape)

        if len(shape) == 4:
            # Handle unbatched (C=6, T=4, H=224, W=224)
            if shape[0] == self.IN_CHANNELS and shape[1] == self.NUM_FRAMES:
                shape = [1] + shape

        if len(shape) != 5:
            return (
                False,
                f"Expected 5D tensor (B, C=6, T=4, H=224, W=224), got {len(shape)}D shape {shape}",
            )

        b, c, t, h, w = shape
        if c != self.IN_CHANNELS:
            return (
                False,
                f"Prithvi requires exactly 6 spectral channels {self.REQUIRED_BANDS}. Got {c} channels (shape {shape}).",
            )

        if t != self.NUM_FRAMES:
            return (
                False,
                f"Prithvi requires exactly 4 temporal frames (T={self.NUM_FRAMES}). Got {t} frames (shape {shape}). Use EOPipeline.prepare_temporal_sequence() to construct 4 observations.",
            )

        if h != self.IMAGE_SIZE or w != self.IMAGE_SIZE:
            return (
                False,
                f"Prithvi requires exactly {self.IMAGE_SIZE}x{self.IMAGE_SIZE} spatial tiles. Got {h}x{w}. Use EOPipeline.tile_raster() or center crop first.",
            )

        return True, "Valid input tensor"

    def normalize_channels(self, raw_bands: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """Normalizes 6-band reflectance data using official Prithvi-EO-2.0-300M climatological constants.
        
        Mean: [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
        Std:  [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]
        """
        is_torch = False
        try:
            import torch
            if isinstance(raw_bands, torch.Tensor):
                is_torch = True
                device = raw_bands.device
                dtype = raw_bands.dtype
                means_t = torch.tensor(self.CHANNEL_MEANS, dtype=dtype, device=device)
                stds_t = torch.tensor(self.CHANNEL_STDS, dtype=dtype, device=device)

                if raw_bands.ndim == 5:
                    means_t = means_t.view(1, 6, 1, 1, 1)
                    stds_t = stds_t.view(1, 6, 1, 1, 1)
                elif raw_bands.ndim == 4:
                    means_t = means_t.view(6, 1, 1, 1)
                    stds_t = stds_t.view(6, 1, 1, 1)
                elif raw_bands.ndim == 3:
                    means_t = means_t.view(6, 1, 1)
                    stds_t = stds_t.view(6, 1, 1)

                return (raw_bands - means_t) / stds_t
        except ImportError:
            pass

        if not is_torch:
            raw = np.asarray(raw_bands, dtype=np.float32)
            if raw.ndim == 5:
                means = self.CHANNEL_MEANS.reshape(1, 6, 1, 1, 1)
                stds = self.CHANNEL_STDS.reshape(1, 6, 1, 1, 1)
            elif raw.ndim == 4:
                means = self.CHANNEL_MEANS.reshape(6, 1, 1, 1)
                stds = self.CHANNEL_STDS.reshape(6, 1, 1, 1)
            else:
                means = self.CHANNEL_MEANS.reshape(6, 1, 1)
                stds = self.CHANNEL_STDS.reshape(6, 1, 1)

            norm = (raw - means) / stds
            return norm.astype(np.float32)

    def load_model(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        force_download: bool = False,
    ) -> Tuple[bool, str]:
        """Lazy thread-safe loader for official Prithvi-EO-2.0-300M checkpoint.
        
        Loads weights from HuggingFace checkpoint pinned to official revision
        9eb1b1102806593963daa333bcc491b1c6f8562f. Removes positional embeddings
        from state_dict before strict=False loading as Prithvi generates 3D sincos
        embeddings dynamically.
        """
        with self._lock:
            if self.status == PrithviModelStatus.READY and self.model is not None:
                return True, "Prithvi-EO-2.0-300M already loaded in memory."

            self.status = PrithviModelStatus.LOADING
            try:
                import torch
            except ImportError:
                self.status = PrithviModelStatus.FAILED
                self.last_error = "PyTorch is not installed in the runtime environment."
                logger.warning(self.last_error)
                return False, self.last_error

            try:
                from icarus.workers.eo.prithvi_mae import PrithviMAE
            except ImportError as ex:
                self.status = PrithviModelStatus.FAILED
                self.last_error = f"Failed to import official PrithviMAE architecture: {str(ex)}"
                logger.error(self.last_error)
                return False, self.last_error

            try:
                # Device setup
                if self.device == "cuda" and not torch.cuda.is_available():
                    self.device = "cpu"

                # Checkpoint resolution
                ckpt_file = checkpoint_path
                if ckpt_file is None:
                    try:
                        from huggingface_hub import hf_hub_download
                        logger.info(
                            "Downloading/retrieving Prithvi checkpoint from %s (rev: %s)",
                            self.HF_REPO_ID,
                            self.HF_REVISION,
                        )
                        ckpt_file = hf_hub_download(
                            repo_id=self.HF_REPO_ID,
                            filename=self.CHECKPOINT_FILENAME,
                            revision=self.HF_REVISION,
                            force_download=force_download,
                        )
                    except Exception as hf_err:
                        self.status = PrithviModelStatus.FAILED
                        self.last_error = f"HuggingFace checkpoint download failed: {str(hf_err)}"
                        logger.error(self.last_error)
                        return False, self.last_error

                # Load state dict
                logger.info("Loading Prithvi checkpoint from %s with weights_only=True", ckpt_file)
                state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)

                # Remove positional embedding weights (computed dynamically)
                for key in list(state_dict.keys()):
                    if "pos_embed" in key:
                        del state_dict[key]

                # Instantiate official PrithviMAE backbone
                model = PrithviMAE(
                    img_size=self.OFFICIAL_CONFIG["img_size"],
                    num_frames=self.OFFICIAL_CONFIG["num_frames"],
                    patch_size=tuple(self.OFFICIAL_CONFIG["patch_size"]),
                    in_chans=self.OFFICIAL_CONFIG["in_chans"],
                    embed_dim=self.OFFICIAL_CONFIG["embed_dim"],
                    depth=self.OFFICIAL_CONFIG["depth"],
                    num_heads=self.OFFICIAL_CONFIG["num_heads"],
                    decoder_embed_dim=self.OFFICIAL_CONFIG["decoder_embed_dim"],
                    decoder_depth=self.OFFICIAL_CONFIG["decoder_depth"],
                    decoder_num_heads=self.OFFICIAL_CONFIG["decoder_num_heads"],
                    mlp_ratio=self.OFFICIAL_CONFIG["mlp_ratio"],
                    coords_encoding=self.OFFICIAL_CONFIG["coords_encoding"],
                    coords_scale_learn=self.OFFICIAL_CONFIG["coords_scale_learn"],
                    mask_ratio=self.OFFICIAL_CONFIG["mask_ratio"],
                    norm_pix_loss=self.OFFICIAL_CONFIG["norm_pix_loss"],
                )

                # Load weights
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                logger.info("Prithvi state_dict loaded. Missing: %d, Unexpected: %d", len(missing), len(unexpected))

                param_count = sum(p.numel() for p in model.parameters())
                if param_count != self.PARAMETER_COUNT:
                    logger.warning("Parameter count mismatch: got %d, expected %d", param_count, self.PARAMETER_COUNT)

                model = model.to(self.device)
                model.eval()

                self.model = model
                self.status = PrithviModelStatus.READY
                self.vram_allocated_mb = 2048 if self.device == "cuda" else 0
                self.last_error = None

                return True, f"Prithvi-EO-2.0-300M ready on {self.device} ({param_count:,} params)."

            except Exception as e:
                self.status = PrithviModelStatus.FAILED
                self.last_error = str(e)
                logger.error("Failed to load Prithvi-EO-2.0-300M: %s", str(e))
                return False, f"Model initialization failed: {str(e)}"

    def unload_model(self) -> Tuple[bool, str, int]:
        """Thread-safe unloading of model weights and CUDA VRAM reclamation."""
        with self._lock:
            freed = self.vram_allocated_mb
            self.model = None
            self.vram_allocated_mb = 0
            self.status = PrithviModelStatus.UNLOADED

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            gc.collect()
            logger.info("Prithvi-EO-2.0-300M unloaded from memory. Reclaimed ~%d MB", freed)
            return True, "Model unloaded successfully", freed

    def extract_features(
        self,
        tensor_5d: Union[np.ndarray, Any],
        temporal_coords: Optional[Any] = None,
        location_coords: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Runs genuine feature extraction through pretrained Prithvi-EO-2.0-300M backbone.
        
        Requires validated 5D input: (Batch, Channels=6, Time_Frames=4, Height=224, Width=224).
        Returns full encoder tokens (B, 785, 1024) and patch embeddings (B, 784, 1024).
        """
        valid, msg = self.validate_input_tensor(tensor_5d)
        if not valid:
            return {"error": msg, "status": "invalid_input"}

        if self.status != PrithviModelStatus.READY or self.model is None:
            return {
                "error": "Prithvi-EO-2.0-300M is not loaded. Call load_model() first.",
                "status": "model_unloaded",
            }

        try:
            import torch
        except ImportError:
            return {"error": "PyTorch is required for Prithvi inference.", "status": "missing_dependency"}

        # Prepare PyTorch tensor
        if isinstance(tensor_5d, np.ndarray):
            if tensor_5d.ndim == 4:
                tensor_5d = np.expand_dims(tensor_5d, axis=0)
            tensor_torch = torch.from_numpy(tensor_5d).float().to(self.device)
        elif isinstance(tensor_5d, torch.Tensor):
            tensor_torch = tensor_5d.float().to(self.device)
            if tensor_torch.ndim == 4:
                tensor_torch = tensor_torch.unsqueeze(0)
        else:
            return {"error": f"Unsupported tensor type: {type(tensor_5d)}", "status": "invalid_input"}

        try:
            with torch.inference_mode():
                out = self.model.forward_features(
                    tensor_torch,
                    temporal_coords=temporal_coords,
                    location_coords=location_coords,
                )
                final_features = out[-1] if isinstance(out, (list, tuple)) else out

                # Verify finiteness
                if not bool(torch.all(torch.isfinite(final_features)).item()):
                    return {
                        "error": "Inference produced NaN or Inf tokens.",
                        "status": "numerical_instability",
                    }

                # Separate CLS token and patch embeddings
                cls_token = final_features[:, :1, :]       # (B, 1, 1024)
                patch_tokens = final_features[:, 1:, :]    # (B, 784, 1024)

                cls_summary = {
                    "mean": float(cls_token.mean().item()),
                    "std": float(cls_token.std().item()),
                    "min": float(cls_token.min().item()),
                    "max": float(cls_token.max().item()),
                }

                patch_summary = {
                    "mean": float(patch_tokens.mean().item()),
                    "std": float(patch_tokens.std().item()),
                    "min": float(patch_tokens.min().item()),
                    "max": float(patch_tokens.max().item()),
                }

                batch_size = tensor_torch.shape[0]
                num_patches = patch_tokens.shape[1]
                hidden_dim = patch_tokens.shape[2]

                return {
                    "status": "success",
                    "model_id": self.MODEL_ID,
                    "parameter_count": self.PARAMETER_COUNT,
                    "input_shape": list(tensor_torch.shape),
                    "token_shape": list(final_features.shape),
                    "patch_token_shape": list(patch_tokens.shape),
                    "patch_count": num_patches,
                    "hidden_dimension": hidden_dim,
                    "cls_token_summary": cls_summary,
                    "patch_embedding_summary": patch_summary,
                    "scientific_note": (
                        "Extracted foundational latent embeddings from IBM/NASA Prithvi-EO-2.0-300M. "
                        "Base ViT encoder outputs 1 CLS token + 784 spatio-temporal patch tokens (4 timesteps x 14x14 patches). "
                        "Does not constitute land-cover, flood mapping, or burn-scar segmentation without a trained task head."
                    ),
                }

        except Exception as inf_err:
            logger.error("Prithvi feature extraction error: %s", str(inf_err))
            return {"error": f"Inference execution failed: {str(inf_err)}", "status": "inference_failed"}
