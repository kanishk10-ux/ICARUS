# -*- coding: utf-8 -*-
"""Unit tests for ICARUS Central Orchestrator, LLM Planner, and Report Generator."""
import json
import httpx
import pytest

from icarus.core.worker_registry import WorkerRegistry
from icarus.orchestrator.llm_planner import LLMPlanner, PlannedSubtask, StructuredResearchPlan
from icarus.orchestrator.orchestrator import CentralResearchOrchestrator
from icarus.orchestrator.report_generator import ReportGenerator
from icarus.schemas.base import CapabilityStatus, DomainEnum, TaskStatus, WorkerRuntimeStatus
from icarus.schemas.worker import CapabilityDescriptor, WorkerRegistration


def test_llm_planner_fallback():
    # Without API key, planner should fall back gracefully
    planner = LLMPlanner(api_key=None)
    multi_req, plan, warnings = planner.plan_research(
        "Assess vegetation health and NDVI near the agricultural reserve"
    )

    assert len(plan.subtasks) >= 1
    assert plan.subtasks[0].domain == DomainEnum.EARTH_OBSERVATION
    assert len(multi_req.subtasks) >= 1
    assert any("deterministic" in w.lower() for w in warnings)


def test_llm_planner_mock_groq(monkeypatch):
    mock_groq_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "identified_domains": ["earth_observation", "climate"],
                        "execution_strategy": "parallel",
                        "scientific_rationale": "Cross-domain correlation of temperature and canopy vigor",
                        "hypotheses": ["Temperature warming suppresses NDVI during peak summer"],
                        "potential_limitations": ["Cloud occlusion in optical bands"],
                        "subtasks": [
                            {
                                "domain": "earth_observation",
                                "task": "eo_vegetation_analysis",
                                "subtask_question": "Compute NDVI for region",
                                "required_models": ["prithvi-eo-2.0-300m"],
                                "parameters": {"threshold": 0.2},
                                "scientific_justification": "Determine canopy density",
                            },
                            {
                                "domain": "climate",
                                "task": "climate_trends",
                                "subtask_question": "Analyze decadal temperature trend",
                                "required_models": [],
                                "parameters": {"years": [2010, 2024]},
                                "scientific_justification": "Identify warming baseline",
                            },
                        ],
                    })
                }
            }
        ]
    }

    def mock_post(self, url, **kwargs):
        return httpx.Response(200, json=mock_groq_response)

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    planner = LLMPlanner(api_key="gsk-test-key-mock")
    multi_req, plan, warnings = planner.plan_research(
        "Investigate climate warming impact on agricultural vegetation"
    )

    assert len(plan.subtasks) == 2
    assert plan.subtasks[0].domain == DomainEnum.EARTH_OBSERVATION
    assert plan.subtasks[1].domain == DomainEnum.CLIMATE
    assert len(multi_req.subtasks) == 2


def test_report_generator_deterministic():
    from icarus.schemas.response import MultiDomainResponse, ResearchResponse, TaskMetric

    resp1 = ResearchResponse(
        request_id="req-1",
        task_id="t-1",
        worker_id="worker-eo-1",
        domain=DomainEnum.EARTH_OBSERVATION,
        status=TaskStatus.SUCCESS,
        model_used="prithvi-eo-2.0-300m",
        results={"summary": "Vegetation index average 0.52"},
        metrics=[TaskMetric(name="mean_ndvi", value=0.52, unit="ratio")],
    )

    multi_resp = MultiDomainResponse(
        request_id="req-multi",
        original_question="Assess vegetation",
        status=TaskStatus.SUCCESS,
        subtask_responses=[resp1],
        synthesized_summary="Vegetation assessment complete.",
    )

    generator = ReportGenerator(api_key=None)
    report = generator.generate_report("Assess vegetation", multi_resp)

    assert report["overall_status"] == TaskStatus.SUCCESS
    assert len(report["measurements"]) == 1
    assert report["measurements"][0]["name"] == "mean_ndvi"
    assert "Vegetation index average 0.52" in report["markdown_report"]
    assert "| earth_observation | mean_ndvi | 0.52 | ratio |" in report["markdown_report"]


def test_central_orchestrator_end_to_end_mock(monkeypatch):
    registry = WorkerRegistry()
    cap_eo = CapabilityDescriptor(
        id="eo_vegetation_analysis",
        domain=DomainEnum.EARTH_OBSERVATION,
        name="Vegetation Index",
        status=CapabilityStatus.IMPLEMENTED,
    )
    worker = WorkerRegistration(
        worker_id="worker-eo-1",
        name="ICARUS_WORKER_EO",
        endpoint_url="http://mock-eo.local",
        supported_domains=[DomainEnum.EARTH_OBSERVATION],
        capabilities={"eo_vegetation_analysis": cap_eo},
        runtime_status=WorkerRuntimeStatus.HEALTHY,
    )
    registry.workers[worker.worker_id] = worker

    def mock_post(self, url, **kwargs):
        payload = kwargs.get("json", {})
        return httpx.Response(200, json={
            "request_id": payload.get("request_id", "req-1"),
            "task_id": payload.get("task_id", "t-1"),
            "worker_id": "worker-eo-1",
            "domain": "earth_observation",
            "status": "success",
            "model_used": "prithvi-eo-2.0-300m",
            "results": {"summary": "NDVI analysis computed successfully", "subdomain": "vegetation_analysis"},
            "metrics": [{"name": "mean_ndvi", "value": 0.45, "unit": "dimensionless"}],
            "artifacts": [],
            "visualizations": {},
            "provenance": {"method": "NDVI"},
            "uncertainty": {},
            "limitations": [],
        })

    def mock_get(self, url, **kwargs):
        return httpx.Response(200, json={
            "worker_id": "worker-eo-1",
            "status": "healthy",
            "uptime_seconds": 500.0,
            "gpu_available": True,
            "gpu_vram_total_mb": 16000,
            "gpu_vram_free_mb": 14000,
            "loaded_models": ["prithvi-eo-2.0-300m"],
            "active_tasks": 0,
        })

    monkeypatch.setattr(httpx.Client, "post", mock_post)
    monkeypatch.setattr(httpx.Client, "get", mock_get)

    orchestrator = CentralResearchOrchestrator(
        worker_registry=registry,
        groq_api_key=None,  # Tests deterministic fallback end-to-end
    )

    report, multi_resp, plan = orchestrator.execute_research(
        "Assess satellite vegetation vigor across agricultural fields"
    )

    assert multi_resp.status == TaskStatus.SUCCESS
    assert len(report["measurements"]) == 1
    assert report["measurements"][0]["name"] == "mean_ndvi"
    assert report["measurements"][0]["value"] == 0.45
    assert len(multi_resp.subtask_responses) == 1
    assert multi_resp.subtask_responses[0].status == TaskStatus.SUCCESS

