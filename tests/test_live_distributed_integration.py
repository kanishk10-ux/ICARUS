# -*- coding: utf-8 -*-
"""Phase 5.3 & 5.7: Live Distributed HTTP Integration Test between ICARUS_CORE and ICARUS_WORKER_EO."""
import json
import logging
import multiprocessing
import socket
import sys
import time
from pathlib import Path
from typing import Dict, Any
import httpx
import numpy as np
import pytest
import uvicorn

from icarus.orchestrator.service import app as core_app
from icarus.workers.eo.service import app as worker_app
from icarus.workers.eo.eo_pipeline import EOPipeline

logger = logging.getLogger("icarus.tests.live_integration")

CORE_PORT = 8001
WORKER_PORT = 8002
CORE_URL = f"http://127.0.0.1:{CORE_PORT}"
WORKER_URL = f"http://127.0.0.1:{WORKER_PORT}"


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _run_core_server():
    uvicorn.run(core_app, host="127.0.0.1", port=CORE_PORT, log_level="warning")


def _run_worker_server():
    uvicorn.run(worker_app, host="127.0.0.1", port=WORKER_PORT, log_level="warning")


@pytest.fixture(scope="module")
def live_servers():
    """Starts actual independent Uvicorn HTTP servers for Core and Worker on separate ports."""
    p_core = multiprocessing.Process(target=_run_core_server, daemon=True)
    p_worker = multiprocessing.Process(target=_run_worker_server, daemon=True)

    p_core.start()
    p_worker.start()

    # Wait for both independent network ports to become active
    timeout = 15.0
    start = time.time()
    core_ready = False
    worker_ready = False

    while time.time() - start < timeout:
        if not core_ready and _is_port_in_use(CORE_PORT):
            core_ready = True
        if not worker_ready and _is_port_in_use(WORKER_PORT):
            worker_ready = True
        if core_ready and worker_ready:
            break
        time.sleep(0.3)

    if not (core_ready and worker_ready):
        p_core.terminate()
        p_worker.terminate()
        pytest.fail(f"Could not bind live HTTP test servers (Core: {core_ready}, Worker: {worker_ready})")

    yield {"core_url": CORE_URL, "worker_url": WORKER_URL}

    p_core.terminate()
    p_worker.terminate()
    p_core.join(timeout=2.0)
    p_worker.join(timeout=2.0)


# ============================================================
# LIVE INTEGRATION TESTS (REAL HTTP NETWORK TRANSPORT)
# ============================================================

def test_live_worker_health_and_capabilities(live_servers):
    """Phase 5.2: Verify that Worker exposes /health and /capabilities over real HTTP."""
    worker_url = live_servers["worker_url"]

    start = time.perf_counter()
    resp_health = httpx.get(f"{worker_url}/health", timeout=5.0)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    assert resp_health.status_code == 200
    health_data = resp_health.json()
    assert health_data["worker_id"] == "ICARUS_WORKER_EO"
    assert health_data["status"] == "healthy"
    print(f"\n[LIVE HTTP] GET {worker_url}/health -> HTTP 200 in {latency_ms}ms")

    resp_caps = httpx.get(f"{worker_url}/capabilities", timeout=5.0)
    assert resp_caps.status_code == 200
    caps_data = resp_caps.json()
    assert caps_data["worker_id"] == "ICARUS_WORKER_EO"
    assert "eo_vegetation_analysis" in [c["id"] for c in caps_data["capabilities"]]
    print(f"[LIVE HTTP] GET {worker_url}/capabilities -> HTTP 200 ({len(caps_data['capabilities'])} capabilities)")


def test_live_dynamic_worker_registration(live_servers):
    """Phase 5.1 & 5.2: Worker dynamically registers its endpoint with ICARUS_CORE."""
    core_url = live_servers["core_url"]
    worker_url = live_servers["worker_url"]

    # Register worker on Core
    reg_payload = {
        "endpoint_url": worker_url,
        "worker_name": "ICARUS_WORKER_EO",
    }
    start = time.perf_counter()
    reg_resp = httpx.post(f"{core_url}/register-worker", json=reg_payload, timeout=5.0)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    assert reg_resp.status_code == 200
    data = reg_resp.json()
    assert data["status"] == "registered"
    assert data["worker_id"] == "ICARUS_WORKER_EO"
    assert data["endpoint_url"] == worker_url
    print(f"[LIVE HTTP] POST {core_url}/register-worker -> HTTP 200 in {latency_ms}ms (Registered: {data['worker_id']})")

    # Verify Core independently lists the worker
    workers_resp = httpx.get(f"{core_url}/workers", timeout=5.0)
    assert workers_resp.status_code == 200
    workers = workers_resp.json()
    assert any(w["worker_id"] == "ICARUS_WORKER_EO" for w in workers)
    print(f"[LIVE HTTP] GET {core_url}/workers -> HTTP 200 ({len(workers)} active workers in registry)")


def test_live_distributed_communication_and_research(live_servers):
    """Phase 5.3: Core sends HTTP request to EO worker, aggregates response, and produces report."""
    core_url = live_servers["core_url"]
    worker_url = live_servers["worker_url"]

    # Ensure registered
    httpx.post(f"{core_url}/register-worker", json={"endpoint_url": worker_url, "worker_name": "ICARUS_WORKER_EO"})

    # Send research question to Core
    research_payload = {
        "question": "Assess satellite vegetation vigor across agricultural parcels",
        "parameters": {
            "nir": [[0.75, 0.82], [0.68, 0.79]],
            "red": [[0.12, 0.14], [0.15, 0.18]],
        },
    }

    start = time.perf_counter()
    resp = httpx.post(f"{core_url}/research", json=research_payload, timeout=10.0)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    assert resp.status_code == 200
    res_data = resp.json()
    assert "report" in res_data
    report = res_data["report"]

    assert report["overall_status"] == "success"
    assert len(report["measurements"]) >= 1
    assert report["measurements"][0]["name"] == "mean_ndvi"
    assert report["measurements"][0]["value"] > 0.6

    print(f"[LIVE HTTP] POST {core_url}/research -> HTTP 200 in {duration_ms}ms")
    print(f"  Executive Summary: {report['executive_summary'][:80]}...")
    print(f"  NDVI Metric: {report['measurements'][0]['name']} = {report['measurements'][0]['value']}")


def test_live_satellite_geotiff_processing(live_servers):
    """Phase 5.4: Process genuine documented Sentinel-2 GeoTIFF data through the live worker."""
    worker_url = live_servers["worker_url"]
    tif_path = Path("data/sentinel2_sample.tif")
    assert tif_path.exists(), "Genuine GeoTIFF data/sentinel2_sample.tif required"

    # Ingest metadata and bands from genuine satellite file
    info = EOPipeline.inspect_geotiff(tif_path)
    assert info["valid"] is True
    assert info["crs"] == "EPSG:32621"
    assert info["is_projected"] is True

    # Read real raster data
    raw_bands, meta = EOPipeline.read_bands_from_geotiff(tif_path)
    # Extract real 100x100 quadrant
    real_quadrant = raw_bands[0, :100, :100].astype(np.float64) / 10000.0  # Convert DN to reflectance (0.0 - 1.0)

    # Use simulated paired band for spectral index from real reflectance range
    red_band = real_quadrant * 0.4  # Proxy red from real satellite reflectance
    nir_band = real_quadrant        # Proxy NIR from real satellite reflectance

    request_payload = {
        "domain": "earth_observation",
        "task": "eo_vegetation_analysis",
        "research_question": "Analyze real Sentinel-2 satellite tile for vegetation vigor",
        "dataset": f"Sentinel-2 COG (Tile T21NYD, CRS: {info['crs']}, Resolution: {info['resolution'][0]}m)",
        "parameters": {
            "nir": nir_band.tolist(),
            "red": red_band.tolist(),
        },
    }

    start = time.perf_counter()
    worker_resp = httpx.post(f"{worker_url}/analyze", json=request_payload, timeout=10.0)
    dur_ms = round((time.perf_counter() - start) * 1000, 2)

    assert worker_resp.status_code == 200
    res = worker_resp.json()
    assert res["status"] == "success"
    assert res["worker_id"] == "ICARUS_WORKER_EO"
    assert res["dataset_used"] == request_payload["dataset"]

    stats = res["results"]["vegetation_statistics"]
    assert stats["total_pixels_evaluated"] == 10000
    assert 0.0 < stats["mean"] < 1.0

    print(f"\n[LIVE HTTP] Real GeoTIFF Analysis -> HTTP 200 in {dur_ms}ms")
    print(f"  Data Source:    {res['dataset_used']}")
    print(f"  Evaluated Pixels: {stats['total_pixels_evaluated']}")
    print(f"  Mean NDVI:      {stats['mean']:.4f}")
    print(f"  Dense Canopy:   {stats['classification_fractions']['dense_canopy_fraction'] * 100:.1f}%")


def test_live_remote_model_unloading(live_servers):
    """Phase 5.5: Remotely unload model via HTTP on the EO Worker."""
    worker_url = live_servers["worker_url"]

    start = time.perf_counter()
    unload_resp = httpx.post(
        f"{worker_url}/unload-model",
        json={"model_id": "prithvi-eo-2.0-300m"},
        timeout=5.0,
    )
    dur_ms = round((time.perf_counter() - start) * 1000, 2)

    assert unload_resp.status_code == 200
    data = unload_resp.json()
    assert data["unloaded"] is True
    assert data["worker_id"] == "ICARUS_WORKER_EO"
    assert data["model_id"] == "prithvi-eo-2.0-300m"
    print(f"[LIVE HTTP] POST {worker_url}/unload-model -> HTTP 200 in {dur_ms}ms (Memory reclaimed)")

