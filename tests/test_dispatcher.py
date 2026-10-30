# -*- coding: utf-8 -*-
"""Unit tests for ICARUS Task Dispatcher."""
import httpx
import pytest

from icarus.core.dispatcher import TaskDispatcher
from icarus.core.worker_registry import WorkerRegistry
from icarus.schemas.base import CapabilityStatus, DomainEnum, TaskStatus, WorkerRuntimeStatus
from icarus.schemas.request import MultiDomainRequest, ResearchRequest
from icarus.schemas.response import ResearchResponse
from icarus.schemas.worker import CapabilityDescriptor, WorkerRegistration


def test_dispatcher_no_worker_available():
    registry = WorkerRegistry()
    dispatcher = TaskDispatcher(registry)

    req = ResearchRequest(
        domain=DomainEnum.ATMOSPHERE,
        task="atmos_wind_analysis",
        research_question="Measure wind vectors",
    )

    resp = dispatcher.dispatch(req)
    assert resp.status == TaskStatus.UNAVAILABLE
    assert resp.error is not None
    assert resp.error["code"] == "WORKER_UNAVAILABLE"
    assert "No active worker registered" in resp.results["error"]


def test_dispatcher_successful_dispatch(monkeypatch):
    registry = WorkerRegistry()
    cap = CapabilityDescriptor(
        id="eo_vegetation_analysis",
        domain=DomainEnum.EARTH_OBSERVATION,
        name="Vegetation Index",
        status=CapabilityStatus.IMPLEMENTED,
    )
    worker = WorkerRegistration(
        worker_id="worker-eo-1",
        name="ICARUS_WORKER_EO",
        endpoint_url="http://mock-eo-worker.local",
        supported_domains=[DomainEnum.EARTH_OBSERVATION],
        capabilities={"eo_vegetation_analysis": cap},
        runtime_status=WorkerRuntimeStatus.HEALTHY,
    )
    registry.workers[worker.worker_id] = worker

    mock_resp_payload = {
        "request_id": "req-1",
        "task_id": "task-1",
        "worker_id": "worker-eo-1",
        "domain": "earth_observation",
        "status": "success",
        "model_used": "prithvi-eo-2.0-300m",
        "results": {"summary": "Analysis completed"},
        "metrics": [{"name": "mean_ndvi", "value": 0.48, "unit": "ratio"}],
        "artifacts": [],
        "visualizations": {},
        "provenance": {"method": "NDVI"},
        "uncertainty": {},
        "limitations": [],
    }

    def mock_post(self, url, **kwargs):
        return httpx.Response(200, json=mock_resp_payload)

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    dispatcher = TaskDispatcher(registry)
    req = ResearchRequest(
        request_id="req-1",
        task_id="task-1",
        domain=DomainEnum.EARTH_OBSERVATION,
        task="eo_vegetation_analysis",
        research_question="Calculate NDVI",
    )

    resp = dispatcher.dispatch(req)
    assert resp.status == TaskStatus.SUCCESS
    assert resp.worker_id == "worker-eo-1"
    assert len(resp.metrics) == 1
    assert resp.metrics[0].value == 0.48


def test_dispatcher_multi_domain(monkeypatch):
    registry = WorkerRegistry()
    # Register 2 workers for 2 domains
    cap_eo = CapabilityDescriptor(id="eo_vegetation_analysis", domain=DomainEnum.EARTH_OBSERVATION, name="EO")
    worker_eo = WorkerRegistration(
        worker_id="worker-eo-1",
        name="ICARUS_WORKER_EO",
        endpoint_url="http://mock-eo.local",
        supported_domains=[DomainEnum.EARTH_OBSERVATION],
        capabilities={"eo_vegetation_analysis": cap_eo},
    )
    registry.workers[worker_eo.worker_id] = worker_eo

    cap_clim = CapabilityDescriptor(id="climate_trends", domain=DomainEnum.CLIMATE, name="Climate")
    worker_clim = WorkerRegistration(
        worker_id="worker-clim-1",
        name="ICARUS_WORKER_2",
        endpoint_url="http://mock-clim.local",
        supported_domains=[DomainEnum.CLIMATE],
        capabilities={"climate_trends": cap_clim},
    )
    registry.workers[worker_clim.worker_id] = worker_clim

    def mock_post(self, url, **kwargs):
        payload = kwargs.get("json") or {}
        dom = payload.get("domain")
        return httpx.Response(200, json={
            "request_id": payload.get("request_id", "req"),
            "task_id": payload.get("task_id", "tsk"),
            "worker_id": "mock-worker",
            "domain": dom,
            "status": "success",
            "results": {"summary": f"Completed {dom} analysis"},
            "metrics": [],
            "artifacts": [],
            "visualizations": {},
            "provenance": {},
            "uncertainty": {},
            "limitations": [],
        })

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    dispatcher = TaskDispatcher(registry)
    multi_req = MultiDomainRequest(
        original_question="Assess regional climate impact on vegetation",
        subtasks=[
            ResearchRequest(domain=DomainEnum.EARTH_OBSERVATION, task="eo_vegetation_analysis", research_question="Vegetation"),
            ResearchRequest(domain=DomainEnum.CLIMATE, task="climate_trends", research_question="Climate trends"),
        ],
    )

    multi_resp = dispatcher.dispatch_multi(multi_req)
    assert multi_resp.status == TaskStatus.SUCCESS
    assert len(multi_resp.subtask_responses) == 2
    assert "Completed earth_observation analysis" in multi_resp.synthesized_summary
    assert "Completed climate analysis" in multi_resp.synthesized_summary
