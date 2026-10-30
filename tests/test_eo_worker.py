# -*- coding: utf-8 -*-
"""Comprehensive verification tests for Earth Observation Worker (ICARUS_WORKER_EO)."""
import numpy as np
import pytest
from starlette.testclient import TestClient

from icarus.core.worker_registry import WorkerRegistry
from icarus.orchestrator.orchestrator import CentralResearchOrchestrator
from icarus.schemas.base import DomainEnum, TaskStatus
from icarus.schemas.request import ResearchRequest
from icarus.workers.eo.eo_pipeline import EOPipeline
from icarus.workers.eo.prithvi_adapter import PrithviEOAdapter, PrithviModelStatus
from icarus.workers.eo.service import app


@pytest.fixture
def test_client():
    return TestClient(app)


# ============================================================
# 1. Pipeline & Algorithm Tests
# ============================================================
def test_spectral_algorithms_math():
    # NDVI: (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.6
    nir = np.array([0.8, 0.6])
    red = np.array([0.2, 0.4])
    ndvi = EOPipeline.compute_ndvi(nir, red)
    assert abs(ndvi[0] - 0.6) < 1e-5
    assert abs(ndvi[1] - 0.2) < 1e-5

    # NDWI: (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.6667
    green = np.array([0.5])
    nir_water = np.array([0.1])
    ndwi = EOPipeline.compute_ndwi(green, nir_water)
    assert abs(ndwi[0] - (0.4 / 0.6)) < 1e-5

    # NBR: (0.7 - 0.3) / (0.7 + 0.3) = 0.4 / 1.0 = 0.4
    swir = np.array([0.3])
    nbr = EOPipeline.compute_nbr(np.array([0.7]), swir)
    assert abs(nbr[0] - 0.4) < 1e-5


def test_surface_area_crs_guard():
    # Geographic degrees CRS must be rejected
    area_deg = EOPipeline.calculate_surface_area_km2(1000, 10.0, crs="EPSG:4326")
    assert "error" in area_deg
    assert area_deg["area_km2"] is None

    # Projected CRS must calculate valid area
    area_proj = EOPipeline.calculate_surface_area_km2(10000, 10.0, crs="EPSG:32643")
    # 10000 * 100 m² = 1,000,000 m² = 1.0 km²
    assert abs(area_proj["area_km2"] - 1.0) < 1e-5


def test_spatial_tiling_and_reconstruction():
    # Scene: 6 channels, 400 x 400
    scene = np.random.uniform(0.1, 0.9, (6, 400, 400)).astype(np.float32)
    tiles, coords, orig_shape = EOPipeline.tile_raster(scene, tile_size=224, stride=224)

    assert len(tiles) == 4  # (2x2 grid with padding for 400x400)
    assert tiles[0].shape == (6, 224, 224)

    reconstructed = EOPipeline.reconstruct_raster(tiles, coords, orig_shape, tile_size=224)
    assert reconstructed.shape == (6, 400, 400)
    diff = np.max(np.abs(scene[:, :200, :200] - reconstructed[:, :200, :200]))
    assert diff < 1e-5


def test_temporal_sequence_preparation():
    # 4 distinct frames of (6, 224, 224)
    frames = [np.ones((6, 224, 224), dtype=np.float32) * i for i in range(4)]
    seq, prov = EOPipeline.prepare_temporal_sequence(frames)
    assert seq.shape == (6, 4, 224, 224)
    assert prov["temporal_observations"] == 4
    assert prov["temporal_strategy"] == "multi_observation_sequence"

    # Single frame rejection without repeat flag
    single_frame = [np.ones((6, 224, 224), dtype=np.float32)]
    with pytest.raises(ValueError, match="Prithvi requires exactly 4 temporal frames"):
        EOPipeline.prepare_temporal_sequence(single_frame, repeat_single_frame=False)

    # Single frame duplication with repeat flag
    seq_rep, prov_rep = EOPipeline.prepare_temporal_sequence(single_frame, repeat_single_frame=True)
    assert seq_rep.shape == (6, 4, 224, 224)
    assert prov_rep["temporal_strategy"] == "single_observation_repeated_4x"
    assert "temporal_note" in prov_rep


# ============================================================
# 2. Prithvi Adapter Tests
# ============================================================
def test_prithvi_band_and_shape_validation():
    adapter = PrithviEOAdapter(device="cpu")
    assert len(adapter.REQUIRED_BANDS) == 6
    assert adapter.NUM_FRAMES == 4
    assert adapter.PARAMETER_COUNT == 330419712

    # Valid 5D input: (Batch=1, Channels=6, Time=4, H=224, W=224)
    valid_tensor = np.zeros((1, 6, 4, 224, 224), dtype=np.float32)
    valid, msg = adapter.validate_input_tensor(valid_tensor)
    assert valid is True

    # Valid 4D unbatched input: (Channels=6, Time=4, H=224, W=224)
    valid_4d = np.zeros((6, 4, 224, 224), dtype=np.float32)
    valid, msg = adapter.validate_input_tensor(valid_4d)
    assert valid is True

    # Invalid channels: 5 bands instead of 6
    invalid_channels = np.zeros((1, 5, 4, 224, 224), dtype=np.float32)
    valid, msg = adapter.validate_input_tensor(invalid_channels)
    assert valid is False
    assert "6 spectral channels" in msg

    # Invalid temporal frames: 3 frames instead of 4
    invalid_temporal = np.zeros((1, 6, 3, 224, 224), dtype=np.float32)
    valid, msg = adapter.validate_input_tensor(invalid_temporal)
    assert valid is False
    assert "4 temporal frames" in msg

    # Invalid dimensions: 100x100 instead of 224x224
    invalid_size = np.zeros((1, 6, 4, 100, 100), dtype=np.float32)
    valid, msg = adapter.validate_input_tensor(invalid_size)
    assert valid is False
    assert "224x224" in msg

    # NaN / Inf rejection
    nan_tensor = np.zeros((1, 6, 4, 224, 224), dtype=np.float32)
    nan_tensor[0, 0, 0, 10, 10] = np.nan
    valid, msg = adapter.validate_input_tensor(nan_tensor)
    assert valid is False
    assert "NaN or Inf" in msg

    inf_tensor = np.zeros((1, 6, 4, 224, 224), dtype=np.float32)
    inf_tensor[0, 0, 0, 10, 10] = np.inf
    valid, msg = adapter.validate_input_tensor(inf_tensor)
    assert valid is False
    assert "NaN or Inf" in msg


def test_prithvi_official_normalization():
    adapter = PrithviEOAdapter(device="cpu")
    # Expected means: [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
    # Expected stds:  [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]
    expected_means = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32)
    expected_stds = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32)

    np.testing.assert_allclose(adapter.CHANNEL_MEANS, expected_means)
    np.testing.assert_allclose(adapter.CHANNEL_STDS, expected_stds)

    # Input identical to mean values should normalize to zeros
    raw = np.zeros((1, 6, 4, 224, 224), dtype=np.float32)
    for c in range(6):
        raw[:, c, :, :, :] = expected_means[c]

    norm = adapter.normalize_channels(raw)
    np.testing.assert_allclose(norm, 0.0, atol=1e-5)


def test_prithvi_model_unloaded_and_lifecycle():
    adapter = PrithviEOAdapter(device="cpu")
    assert adapter.status == PrithviModelStatus.UNLOADED

    # Attempt extraction when unloaded
    dummy_input = np.zeros((1, 6, 4, 224, 224), dtype=np.float32)
    res = adapter.extract_features(dummy_input)
    assert res["status"] == "model_unloaded"
    assert "not loaded" in res["error"]

    # Lifecycle: Unload should be clean even when already unloaded
    unloaded, msg, freed = adapter.unload_model()
    assert unloaded is True
    assert adapter.status == PrithviModelStatus.UNLOADED
    assert adapter.model is None


# ============================================================
# 3. FastAPI Service Endpoint Tests
# ============================================================
def test_worker_health_endpoint(test_client):
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["worker_id"] == "ICARUS_WORKER_EO"
    assert data["status"] == "healthy"
    assert "gpu_available" in data


def test_worker_capabilities_endpoint(test_client):
    resp = test_client.get("/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["worker_id"] == "ICARUS_WORKER_EO"
    assert DomainEnum.EARTH_OBSERVATION in data["supported_domains"]
    cap_ids = [c["id"] for c in data["capabilities"]]
    assert "eo_vegetation_analysis" in cap_ids
    assert "eo_foundation_embeddings" in cap_ids
    assert "eo_flood_mapping" in cap_ids


def test_worker_analyze_ndvi(test_client):
    payload = {
        "domain": "earth_observation",
        "task": "eo_vegetation_analysis",
        "research_question": "Compute vegetation health index",
        "parameters": {
            "nir": [[0.6, 0.7], [0.8, 0.5]],
            "red": [[0.2, 0.1], [0.2, 0.3]],
        },
    }
    resp = test_client.post("/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == TaskStatus.SUCCESS
    assert data["domain"] == "earth_observation"
    assert len(data["metrics"]) >= 1
    assert data["metrics"][0]["name"] == "mean_ndvi"


def test_worker_analyze_prithvi_missing_data_rejection(test_client):
    # Without allow_synthetic_demo and without data payload, request must fail
    payload = {
        "domain": "earth_observation",
        "task": "eo_foundation_embeddings",
        "research_question": "Extract latent representations",
        "allow_synthetic_demo": False,
    }
    resp = test_client.post("/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == TaskStatus.FAILED
    assert data["error"]["code"] == "MISSING_INPUT_DATA"


def test_worker_analyze_prithvi_with_geotiff(test_client):
    # With genuine 6-band GeoTIFF, input validation succeeds (proceeds to model loading)
    payload = {
        "domain": "earth_observation",
        "task": "eo_foundation_embeddings",
        "research_question": "Extract latent representations from genuine GeoTIFF",
        "parameters": {"raster_path": "data/hls_sample_6band.tif"},
        "allow_synthetic_demo": False,
    }
    resp = test_client.post("/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == TaskStatus.FAILED
    # Fails at model loading in environment without PyTorch, NOT at input ingestion
    assert data["error"]["code"] == "MODEL_LOAD_FAILED"


def test_worker_analyze_flood_unavailability_honesty(test_client):
    # Base model does not claim flood mapping without fine-tuned head
    payload = {
        "domain": "earth_observation",
        "task": "eo_flood_mapping",
        "research_question": "Generate flood inundation mask",
    }
    resp = test_client.post("/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == TaskStatus.UNAVAILABLE
    assert "DOWNSTREAM_HEAD_UNAVAILABLE" in data["error"]["code"]


def test_worker_unload_model_endpoint(test_client):
    # Tests both official and legacy model IDs
    resp1 = test_client.post("/unload-model", json={"model_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"})
    assert resp1.status_code == 200
    assert resp1.json()["unloaded"] is True

    resp2 = test_client.post("/unload-model", json={"model_id": "prithvi-eo-2.0-300m"})
    assert resp2.status_code == 200
    assert resp2.json()["unloaded"] is True


# ============================================================
# 4. End-to-End Orchestrator Dispatch to EO Worker
# ============================================================
def test_end_to_end_orchestrator_to_eo_worker(monkeypatch):
    from icarus.schemas.worker import CapabilityDescriptor, WorkerRegistration

    registry = WorkerRegistry()
    client = TestClient(app)

    # Register local worker directly
    cap = CapabilityDescriptor(
        id="eo_vegetation_analysis",
        domain=DomainEnum.EARTH_OBSERVATION,
        name="Vegetation Index",
    )
    worker = WorkerRegistration(
        worker_id="ICARUS_WORKER_EO",
        name="ICARUS_WORKER_EO",
        endpoint_url="http://eo-worker.internal",
        supported_domains=[DomainEnum.EARTH_OBSERVATION],
        capabilities={"eo_vegetation_analysis": cap},
        available_models=["ibm-nasa-geospatial/Prithvi-EO-2.0-300M", "prithvi-eo-2.0-300m"],
    )
    registry.workers[worker.worker_id] = worker

    # Mock httpx in dispatcher to route to FastAPI TestClient
    import httpx
    def mock_post(self, url, **kwargs):
        payload = kwargs.get("json", {})
        res = client.post("/analyze", json=payload)
        return httpx.Response(res.status_code, json=res.json())

    def mock_get(self, url, **kwargs):
        res = client.get("/health")
        return httpx.Response(res.status_code, json=res.json())

    monkeypatch.setattr(httpx.Client, "post", mock_post)
    monkeypatch.setattr(httpx.Client, "get", mock_get)

    orchestrator = CentralResearchOrchestrator(
        worker_registry=registry,
        groq_api_key=None,
    )

    report, multi_resp, plan = orchestrator.execute_research(
        "Analyze vegetation index across agricultural zone",
        parameters={
            "nir": [[0.7, 0.8], [0.65, 0.75]],
            "red": [[0.12, 0.15], [0.14, 0.18]],
        },
    )

    assert multi_resp.status == TaskStatus.SUCCESS
    assert len(report["measurements"]) >= 1
    assert report["measurements"][0]["name"] == "mean_ndvi"
    assert report["measurements"][0]["value"] > 0.5
    assert len(multi_resp.subtask_responses) == 1
    assert multi_resp.subtask_responses[0].worker_id == "ICARUS_WORKER_EO"
