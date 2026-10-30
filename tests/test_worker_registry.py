# -*- coding: utf-8 -*-
"""Unit tests for ICARUS Worker Registry and Health Checker."""
from datetime import datetime, timezone
import httpx
import pytest

from icarus.core.health_checker import HealthChecker
from icarus.core.worker_registry import WorkerRegistry
from icarus.schemas.base import CapabilityStatus, DomainEnum, WorkerRuntimeStatus
from icarus.schemas.worker import (
    CapabilityDescriptor,
    HealthCheckResponse,
    WorkerCapabilitiesResponse,
    WorkerRegistration,
)


def test_manual_worker_registration_and_filtering():
    registry = WorkerRegistry()

    cap_ndvi = CapabilityDescriptor(
        id="eo_vegetation_analysis",
        domain=DomainEnum.EARTH_OBSERVATION,
        name="Vegetation Index Analysis",
        status=CapabilityStatus.IMPLEMENTED,
        required_models=["prithvi-eo-2.0-300m"],
    )

    worker = WorkerRegistration(
        worker_id="worker-eo-node-1",
        name="ICARUS_WORKER_EO",
        endpoint_url="https://eo-worker.ngrok-free.app",
        supported_domains=[DomainEnum.EARTH_OBSERVATION],
        capabilities={"eo_vegetation_analysis": cap_ndvi},
        available_models=["prithvi-eo-2.0-300m"],
        loaded_models=[],
        gpu_available=True,
        gpu_vram_total_mb=16000,
        gpu_vram_free_mb=15000,
        runtime_status=WorkerRuntimeStatus.HEALTHY,
    )

    registry.workers[worker.worker_id] = worker

    # Find worker by domain and task
    found = registry.find_workers_for_task(
        domain=DomainEnum.EARTH_OBSERVATION,
        task="eo_vegetation_analysis",
    )
    assert len(found) == 1
    assert found[0].worker_id == "worker-eo-node-1"

    # Find worker requiring an unavailable model
    not_found = registry.find_workers_for_task(
        domain=DomainEnum.EARTH_OBSERVATION,
        task="eo_vegetation_analysis",
        required_models=["nonexistent-model-xyz"],
    )
    assert len(not_found) == 0

    # Unrelated domain should return empty
    not_found_domain = registry.find_workers_for_task(
        domain=DomainEnum.ASTRONOMY,
        task="astro_stellar_classification",
    )
    assert len(not_found_domain) == 0


def test_health_checker_mock_healthy(monkeypatch):
    registry = WorkerRegistry()
    worker = WorkerRegistration(
        worker_id="worker-eo-1",
        name="ICARUS_WORKER_EO",
        endpoint_url="http://mock-worker.local",
        supported_domains=[DomainEnum.EARTH_OBSERVATION],
        runtime_status=WorkerRuntimeStatus.HEALTHY,
    )
    registry.workers[worker.worker_id] = worker

    mock_health = {
        "worker_id": "worker-eo-1",
        "status": "healthy",
        "uptime_seconds": 1240.5,
        "gpu_available": True,
        "gpu_vram_total_mb": 16000,
        "gpu_vram_free_mb": 12000,
        "loaded_models": ["prithvi-eo-2.0-300m"],
        "active_tasks": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    def mock_get(self, url, **kwargs):
        return httpx.Response(200, json=mock_health)

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    checker = HealthChecker(registry, timeout_seconds=2.0)
    res = checker.check_worker("worker-eo-1")
    assert res is not None
    assert res.status == WorkerRuntimeStatus.HEALTHY
    assert worker.gpu_vram_free_mb == 12000
    assert worker.loaded_models == ["prithvi-eo-2.0-300m"]
    assert worker.consecutive_failures == 0
