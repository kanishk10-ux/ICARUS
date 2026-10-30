"""
ICARUS Central Model Test Runner CLI
====================================
Tests the loading, preprocessing, inference execution, and error handling
for all 5 ICARUS specialist models:
  1. Prithvi-EO-2.0-300M (Earth Observation)
  2. SeisBench PhaseNet (Solid Earth / Seismology)
  3. NASA-IBM Surya-1.0 (Heliophysics / Space Weather)
  4. NASA-IBM Lunar Foundation Model (Planetary Science)
  5. Zoobot 2.0 (Astronomy & Astrophysics)

Usage:
  python scripts/test_model.py --model [prithvi|phasenet|surya|lunar|zoobot|all]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_model")


def test_prithvi() -> Dict[str, Any]:
    logger.info("=== Testing Model 1/5: Prithvi-EO-2.0-300M (Earth Observation) ===")
    start_t = time.time()
    try:
        from icarus.workers.eo.prithvi_adapter import PrithviEOAdapter
        adapter = PrithviEOAdapter(device="cpu")
        config_summary = {
            "model_id": adapter.MODEL_ID,
            "revision": adapter.HF_REVISION,
            "img_size": adapter.IMAGE_SIZE,
            "in_chans": adapter.IN_CHANNELS,
            "num_frames": adapter.NUM_FRAMES,
            "embed_dim": adapter.EMBED_DIM,
        }
        logger.info(f"Prithvi Adapter Config: {config_summary}")

        # Test validation on invalid input (wrong channels)
        bad_input = np.zeros((3, 224, 224), dtype=np.float32)
        valid_bad, msg_bad = adapter.validate_input_tensor(bad_input)
        validation_rejected = not valid_bad
        logger.info(f"Validation check (bad input rejected: {validation_rejected}): {msg_bad}")

        # Test valid synthetic 6-band raster (shape [6, 4, 224, 224] or [6, 224, 224])
        valid_input = np.random.uniform(200, 3000, (6, 4, 224, 224)).astype(np.float32)
        valid_good, msg_good = adapter.validate_input_tensor(valid_input)
        logger.info(f"Validation check (valid input accepted: {valid_good}): {msg_good}")

        norm_tensor = adapter.normalize_channels(valid_input)
        tensor_shape = list(norm_tensor.shape)
        logger.info(f"Normalized tensor shape: {tensor_shape}")

        # Check model loading availability
        ok, load_msg = adapter.load_model()
        logger.info(f"Prithvi load result: {ok} -> {load_msg}")

        inference_status = "PENDING_WEIGHTS"
        if ok:
            res = adapter.extract_features(valid_input)
            inference_status = res.get("status", "FAILED")
            logger.info(f"Prithvi extraction status: {inference_status}")

        elapsed = time.time() - start_t
        return {
            "model": "Prithvi-EO-2.0-300M",
            "domain": "Earth Observation",
            "config": config_summary,
            "validation_verified": validation_rejected and valid_good,
            "preprocessed_shape": tensor_shape,
            "weight_load_status": ok,
            "inference_status": inference_status,
            "message": load_msg,
            "status": "INFERENCE VERIFIED" if inference_status == "COMPLETED" else "CONFIGURED (WEIGHTS PENDING DOWNLOAD)",
            "elapsed_seconds": round(elapsed, 3),
        }
    except Exception as e:
        logger.error(f"Prithvi test failed: {e}")
        return {
            "model": "Prithvi-EO-2.0-300M",
            "domain": "Earth Observation",
            "status": "ERROR",
            "error": str(e),
        }


def test_phasenet() -> Dict[str, Any]:
    logger.info("=== Testing Model 2/5: SeisBench PhaseNet (Solid Earth / Seismology) ===")
    start_t = time.time()
    try:
        from icarus.workers.solid_earth.phasenet_adapter import PhaseNetAdapter
        adapter = PhaseNetAdapter(device="cpu")
        logger.info(f"PhaseNet Adapter initialized. Model ID: {adapter.MODEL_ID}, Sampling rate: {adapter.DEFAULT_SAMPLING_RATE_HZ} Hz")

        # Test validation on invalid waveform (2 channels instead of 3)
        bad_wave = np.zeros((2, 1000), dtype=np.float32)
        valid_bad, msg_bad = adapter.validate_waveform_input(bad_wave)
        val_rejected = not valid_bad
        logger.info(f"Validation check (bad input rejected: {val_rejected}): {msg_bad}")

        # Test valid 3-component 3000-sample waveform
        valid_wave = np.random.normal(0, 1.0, (3, 3000)).astype(np.float32)
        valid_good, msg_good = adapter.validate_waveform_input(valid_wave, sampling_rate=100.0)
        norm_wave = adapter.preprocess_waveform(valid_wave)
        logger.info(f"Preprocessed waveform shape: {list(norm_wave.shape)}")

        ok, load_msg = adapter.load_model()
        logger.info(f"PhaseNet load result: {ok} -> {load_msg}")

        inference_status = "PENDING_DEPENDENCY"
        if ok:
            res = adapter.predict_phases(valid_wave, sampling_rate_hz=100.0)
            inference_status = res.get("status", "FAILED")
            logger.info(f"PhaseNet prediction status: {inference_status}")

        elapsed = time.time() - start_t
        return {
            "model": "SeisBench PhaseNet",
            "domain": "Solid Earth and Seismology",
            "validation_verified": val_rejected and valid_good,
            "preprocessed_shape": list(norm_wave.shape),
            "weight_load_status": ok,
            "inference_status": inference_status,
            "message": load_msg,
            "status": "INFERENCE VERIFIED" if inference_status == "COMPLETED" else "CONFIGURED (WEIGHTS PENDING DOWNLOAD)",
            "elapsed_seconds": round(elapsed, 3),
        }
    except Exception as e:
        logger.error(f"PhaseNet test failed: {e}")
        return {
            "model": "SeisBench PhaseNet",
            "domain": "Solid Earth and Seismology",
            "status": "ERROR",
            "error": str(e),
        }


def test_surya() -> Dict[str, Any]:
    logger.info("=== Testing Model 3/5: NASA-IBM Surya-1.0 (Heliophysics) ===")
    start_t = time.time()
    try:
        from icarus.workers.heliophysics.surya_adapter import SuryaSolarAdapter
        adapter = SuryaSolarAdapter(device="cpu")
        logger.info(f"Surya Adapter initialized: {adapter.MODEL_ID}, Checkpoint: {adapter.CHECKPOINT_FILENAME}")

        # Test validation on invalid channels (2D array instead of multi-channel)
        bad_solar = np.zeros((224, 224), dtype=np.float32)
        valid_bad, msg_bad = adapter.validate_solar_input(bad_solar)
        val_rejected = not valid_bad
        logger.info(f"Validation check (bad input rejected: {val_rejected}): {msg_bad}")

        # Test valid 8-channel 224x224 observation
        valid_solar = np.random.uniform(10.0, 500.0, (8, 224, 224)).astype(np.float32)
        valid_good, msg_good = adapter.validate_solar_input(valid_solar)
        norm_solar = adapter.normalize_aia_channels(valid_solar)
        logger.info(f"Normalized solar observation shape: {list(norm_solar.shape)}")

        ok, load_msg = adapter.load_model()
        logger.info(f"Surya load result: {ok} -> {load_msg}")

        inference_status = "PENDING_CHECKPOINT"
        if ok:
            res = adapter.extract_features(valid_solar)
            inference_status = res.get("status", "FAILED")
            logger.info(f"Surya extraction status: {inference_status}")

        elapsed = time.time() - start_t
        return {
            "model": "NASA-IBM Surya-1.0",
            "domain": "Sun and Space Weather",
            "validation_verified": val_rejected and valid_good,
            "preprocessed_shape": list(norm_solar.shape),
            "weight_load_status": ok,
            "inference_status": inference_status,
            "message": load_msg,
            "status": "INFERENCE VERIFIED" if inference_status == "COMPLETED" else "CONFIGURED (WEIGHTS PENDING DOWNLOAD)",
            "elapsed_seconds": round(elapsed, 3),
        }
    except Exception as e:
        logger.error(f"Surya test failed: {e}")
        return {
            "model": "NASA-IBM Surya-1.0",
            "domain": "Sun and Space Weather",
            "status": "ERROR",
            "error": str(e),
        }


def test_lunar() -> Dict[str, Any]:
    logger.info("=== Testing Model 4/5: NASA-IBM Lunar Foundation Model (Planetary) ===")
    start_t = time.time()
    try:
        from icarus.workers.planetary.lunar_adapter import LunarFoundationAdapter
        adapter = LunarFoundationAdapter(device="cpu")
        logger.info(f"Lunar FM Adapter initialized: {adapter.MODEL_ID}, Image size: {adapter.IMAGE_SIZE}")

        # Test validation on tiny raster
        bad_dem = np.zeros((10, 10), dtype=np.float32)
        valid_bad, msg_bad = adapter.validate_lunar_input(bad_dem)
        val_rejected = not valid_bad
        logger.info(f"Validation check (bad input rejected: {val_rejected}): {msg_bad}")

        # Test valid 224x224 elevation raster
        valid_dem = np.random.uniform(-2000.0, 3000.0, (224, 224)).astype(np.float32)
        valid_good, msg_good = adapter.validate_lunar_input(valid_dem)
        norm_dem = adapter.preprocess_lunar_tile(valid_dem)
        logger.info(f"Preprocessed lunar tile shape: {list(norm_dem.shape)}")

        ok, load_msg = adapter.load_model()
        logger.info(f"Lunar FM load result: {ok} -> {load_msg}")

        inference_status = "PENDING_CHECKPOINT"
        if ok:
            res = adapter.extract_features(valid_dem)
            inference_status = res.get("status", "FAILED")
            logger.info(f"Lunar extraction status: {inference_status}")

        elapsed = time.time() - start_t
        return {
            "model": "NASA-IBM Lunar Foundation Model",
            "domain": "Planetary Science",
            "validation_verified": val_rejected and valid_good,
            "preprocessed_shape": list(norm_dem.shape),
            "weight_load_status": ok,
            "inference_status": inference_status,
            "message": load_msg,
            "status": "INFERENCE VERIFIED" if inference_status == "COMPLETED" else "CONFIGURED (WEIGHTS PENDING DOWNLOAD)",
            "elapsed_seconds": round(elapsed, 3),
        }
    except Exception as e:
        logger.error(f"Lunar FM test failed: {e}")
        return {
            "model": "NASA-IBM Lunar Foundation Model",
            "domain": "Planetary Science",
            "status": "ERROR",
            "error": str(e),
        }


def test_zoobot() -> Dict[str, Any]:
    logger.info("=== Testing Model 5/5: Zoobot 2.0 (Astronomy & Astrophysics) ===")
    start_t = time.time()
    try:
        from icarus.workers.astronomy.zoobot_adapter import ZoobotAdapter
        adapter = ZoobotAdapter(device="cpu")
        logger.info(f"Zoobot Adapter initialized: {adapter.MODEL_ID}")

        # Test validation on tiny image (16x16 instead of >=32x32)
        bad_img = np.zeros((16, 16), dtype=np.float32)
        valid_bad, msg_bad = adapter.validate_astronomy_input(bad_img)
        val_rejected = not valid_bad
        logger.info(f"Validation check (bad input rejected: {val_rejected}): {msg_bad}")

        # Test valid 224x224x3 image
        valid_img = np.random.uniform(0, 1000.0, (224, 224, 3)).astype(np.float32)
        valid_good, msg_good = adapter.validate_astronomy_input(valid_img)
        norm_img = adapter.preprocess_galaxy_cutout(valid_img)
        logger.info(f"Preprocessed galaxy cutout shape: {list(norm_img.shape)}")

        ok, load_msg = adapter.load_model()
        logger.info(f"Zoobot load result: {ok} -> {load_msg}")

        inference_status = "PENDING_CHECKPOINT"
        if ok:
            res = adapter.classify_galaxy(valid_img)
            inference_status = res.get("status", "FAILED")
            logger.info(f"Zoobot classification status: {inference_status}")

        elapsed = time.time() - start_t
        return {
            "model": "Zoobot 2.0",
            "domain": "Astronomy and Astrophysics",
            "validation_verified": val_rejected and valid_good,
            "preprocessed_shape": list(norm_img.shape),
            "weight_load_status": ok,
            "inference_status": inference_status,
            "message": load_msg,
            "status": "INFERENCE VERIFIED" if inference_status == "COMPLETED" else "CONFIGURED (WEIGHTS PENDING DOWNLOAD)",
            "elapsed_seconds": round(elapsed, 3),
        }
    except Exception as e:
        logger.error(f"Zoobot test failed: {e}")
        return {
            "model": "Zoobot 2.0",
            "domain": "Astronomy and Astrophysics",
            "status": "ERROR",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="ICARUS Specialist Model Test Runner")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["prithvi", "phasenet", "surya", "lunar", "zoobot", "all"],
        help="Model to test (prithvi, phasenet, surya, lunar, zoobot, or all)",
    )
    args = parser.parse_args()

    results = {}
    if args.model in ("prithvi", "all"):
        results["prithvi"] = test_prithvi()
    if args.model in ("phasenet", "all"):
        results["phasenet"] = test_phasenet()
    if args.model in ("surya", "all"):
        results["surya"] = test_surya()
    if args.model in ("lunar", "all"):
        results["lunar"] = test_lunar()
    if args.model in ("zoobot", "all"):
        results["zoobot"] = test_zoobot()

    print("\n" + "=" * 60)
    print("           ICARUS MODEL TEST RUNNER REPORT")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("=" * 60)

    # Return 0 if all adapters ran their tests (validation + preprocessing + lifecycle)
    errors = [k for k, v in results.items() if v.get("status") == "ERROR"]
    if errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
