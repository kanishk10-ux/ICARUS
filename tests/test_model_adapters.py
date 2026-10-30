"""
Comprehensive Test Suite for ICARUS Specialist Model Adapters
==============================================================
Tests initialization, configuration, input validation, tensor preprocessing,
and lifecycle management for all 5 ICARUS specialist model adapters:
  1. Prithvi-EO-2.0-300M (Earth Observation)
  2. SeisBench PhaseNet (Solid Earth / Seismology)
  3. NASA-IBM Surya-1.0 (Heliophysics / Space Weather)
  4. NASA-IBM Lunar Foundation Model (Planetary Science)
  5. Zoobot 2.0 (Astronomy & Astrophysics)
"""

import numpy as np
import pytest

from icarus.workers.astronomy.zoobot_adapter import ZoobotAdapter
from icarus.workers.eo.prithvi_adapter import PrithviEOAdapter
from icarus.workers.heliophysics.surya_adapter import SuryaSolarAdapter
from icarus.workers.planetary.lunar_adapter import LunarFoundationAdapter
from icarus.workers.solid_earth.phasenet_adapter import PhaseNetAdapter


# =====================================================================
# 1. Prithvi-EO-2.0-300M Tests
# =====================================================================

def test_prithvi_adapter_config_and_validation():
    adapter = PrithviEOAdapter(device="cpu")
    assert adapter.MODEL_ID == "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
    assert adapter.HF_REVISION == "9eb1b1102806593963daa333bcc491b1c6f8562f"
    assert adapter.IMAGE_SIZE == 224
    assert adapter.IN_CHANNELS == 6
    assert adapter.NUM_FRAMES == 4
    assert adapter.EMBED_DIM == 1024

    # Invalid input: 3 channels instead of 6
    bad_raster = np.zeros((3, 224, 224), dtype=np.float32)
    valid, msg = adapter.validate_input_tensor(bad_raster)
    assert not valid
    assert "Expected 5D tensor" in msg

    # Valid input: 6 bands, 4 frames, 224x224
    good_raster = np.random.uniform(500, 2500, (6, 4, 224, 224)).astype(np.float32)
    valid, msg = adapter.validate_input_tensor(good_raster)
    assert valid

    # Normalization check
    norm_tensor = adapter.normalize_channels(good_raster)
    assert norm_tensor.shape == (6, 4, 224, 224)
    # Check that channel 0 mean is roughly centered
    assert np.abs(np.mean(norm_tensor[0])) < 2.0

    # Lifecycle check
    status = adapter.get_status()
    assert "status" in status
    assert status["vram_allocated_mb"] == 0


# =====================================================================
# 2. SeisBench PhaseNet Tests
# =====================================================================

def test_phasenet_adapter_config_and_validation():
    adapter = PhaseNetAdapter(device="cpu")
    assert adapter.MODEL_ID == "seisbench/phasenet"
    assert adapter.DEFAULT_SAMPLING_RATE_HZ == 100.0
    assert adapter.REQUIRED_CHANNELS == 3

    # Invalid input: 1 channel instead of 3
    bad_waveform = np.zeros((1000,), dtype=np.float32)
    valid, msg = adapter.validate_waveform_input(bad_waveform)
    assert not valid
    assert "requires 3-component" in msg

    # Invalid sampling rate: 50 Hz instead of 100 Hz
    valid_wave = np.random.normal(0, 1.0, (3, 3000)).astype(np.float32)
    valid_sr, msg_sr = adapter.validate_waveform_input(valid_wave, sampling_rate=50.0)
    assert not valid_sr
    assert "requires 100 Hz" in msg_sr

    # Valid input: 3 channels, 3000 samples at 100 Hz
    valid, msg = adapter.validate_waveform_input(valid_wave, sampling_rate=100.0)
    assert valid

    # Preprocessing check: zero mean and unit variance per channel
    norm_wave = adapter.preprocess_waveform(valid_wave)
    assert norm_wave.shape == (3, 3000)
    assert np.all(np.abs(np.mean(norm_wave, axis=1)) < 1e-4)
    assert np.all(np.isclose(np.std(norm_wave, axis=1), 1.0, atol=1e-3))

    status = adapter.get_status()
    assert "status" in status


# =====================================================================
# 3. NASA-IBM Surya-1.0 Tests
# =====================================================================

def test_surya_adapter_config_and_validation():
    adapter = SuryaSolarAdapter(device="cpu")
    assert adapter.MODEL_ID == "nasa-ibm-ai4science/Surya-1.0"
    assert adapter.CHECKPOINT_FILENAME == "surya.366m.v1.pt"
    assert adapter.AIA_CHANNELS == ["94", "131", "171", "193", "211", "304", "335", "1600"]

    # Invalid input: 2D array instead of multi-channel
    bad_solar = np.zeros((224, 224), dtype=np.float32)
    valid, msg = adapter.validate_solar_input(bad_solar)
    assert not valid
    assert "Expected multi-channel" in msg

    # Valid input: 8 channels, 224x224
    valid_solar = np.random.uniform(10.0, 500.0, (8, 224, 224)).astype(np.float32)
    valid, msg = adapter.validate_solar_input(valid_solar)
    assert valid

    # Normalization check: log10 and standardization
    norm_solar = adapter.normalize_aia_channels(valid_solar)
    assert norm_solar.shape == (8, 224, 224)
    assert np.all(np.isfinite(norm_solar))

    status = adapter.get_status()
    assert "status" in status


# =====================================================================
# 4. NASA-IBM Lunar Foundation Model Tests
# =====================================================================

def test_lunar_adapter_config_and_validation():
    adapter = LunarFoundationAdapter(device="cpu")
    assert adapter.MODEL_ID == "nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model"
    assert adapter.IMAGE_SIZE == 224
    assert adapter.PATCH_SIZE == 16
    assert adapter.EMBED_DIM == 768

    # Invalid input: 16x16 too small (minimum 64x64)
    bad_dem = np.zeros((16, 16), dtype=np.float32)
    valid, msg = adapter.validate_lunar_input(bad_dem)
    assert not valid
    assert "too small" in msg

    # Valid input: 224x224 elevation grid
    valid_dem = np.random.uniform(-1000.0, 2000.0, (224, 224)).astype(np.float32)
    valid, msg = adapter.validate_lunar_input(valid_dem)
    assert valid

    # Preprocessing check: robust min-max elevation scaling
    norm_dem = adapter.preprocess_lunar_tile(valid_dem)
    assert norm_dem.shape == (224, 224)
    assert np.all(norm_dem >= -1.0) and np.all(norm_dem <= 1.0)

    status = adapter.get_status()
    assert "status" in status


# =====================================================================
# 5. Zoobot 2.0 Tests
# =====================================================================

def test_zoobot_adapter_config_and_validation():
    adapter = ZoobotAdapter(device="cpu")
    assert adapter.MODEL_ID == "mwalmsley/zoobot-encoder-convnext_nano"
    assert adapter.IMAGE_SIZE == 224
    assert adapter.EMBEDDING_DIM == 640

    # Invalid input: 16x16 too small (minimum 32x32)
    bad_cutout = np.zeros((16, 16), dtype=np.float32)
    valid, msg = adapter.validate_astronomy_input(bad_cutout)
    assert not valid
    assert "too small" in msg

    # Valid input: 224x224x3 color cutout
    valid_cutout = np.random.uniform(0.0, 500.0, (224, 224, 3)).astype(np.float32)
    valid, msg = adapter.validate_astronomy_input(valid_cutout)
    assert valid

    # Preprocessing check: arcsinh stretch and percentile scaling
    norm_cutout = adapter.preprocess_galaxy_cutout(valid_cutout)
    assert norm_cutout.shape == (3, 224, 224)
    assert np.all(norm_cutout >= 0.0) and np.all(norm_cutout <= 1.0)

    status = adapter.get_status()
    assert "status" in status
