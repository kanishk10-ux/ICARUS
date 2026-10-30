# -*- coding: utf-8 -*-
"""Unit tests for ICARUS Pydantic schemas."""
import pytest
from icarus.schemas import (
    DomainEnum,
    CapabilityStatus,
    TaskStatus,
    WorkerRuntimeStatus,
    SpatialBounds,
    TemporalRange,
    ArtifactRef,
    ResearchRequest,
    MultiDomainRequest,
    TaskMetric,
    ScientificFinding,
    ExecutionMetadata,
    ResearchResponse,
    MultiDomainResponse,
    CapabilityDescriptor,
    WorkerCapabilitiesResponse,
    HealthCheckResponse,
)


def test_spatial_bounds_validation():
    valid = SpatialBounds(min_lat=10.0, max_lat=20.0, min_lon=70.0, max_lon=80.0)
    assert valid.is_valid() is True

    invalid = SpatialBounds(min_lat=25.0, max_lat=20.0, min_lon=70.0, max_lon=80.0)
    assert invalid.is_valid() is False


def test_research_request_defaults():
    req = ResearchRequest(
        domain=DomainEnum.EARTH_OBSERVATION,
        task="vegetation_analysis",
        research_question="Assess NDVI across agricultural sector",
    )
    assert req.request_id is not None
    assert req.task_id is not None
    assert req.domain == "earth_observation"
    assert req.timeout_seconds == 120.0
    assert req.allow_synthetic_demo is False


def test_research_response_legacy_conversion():
    resp = ResearchResponse(
        request_id="req-123",
        task_id="tsk-456",
        worker_id="worker-eo-1",
        domain=DomainEnum.EARTH_OBSERVATION,
        status=TaskStatus.SUCCESS,
        model_used="prithvi-eo-2.0-300m",
        results={"summary": "Vegetation index computed successfully", "subdomain": "vegetation_analysis"},
        metrics=[TaskMetric(name="mean_ndvi", value=0.55, unit="dimensionless")],
        provenance={"methodology": ["Normalized difference of NIR and Red"]},
        limitations=["Atmospheric interference not fully removed"],
    )

    legacy = resp.to_legacy_dict()
    assert legacy["domain"] == "earth_observation"
    assert legacy["status"] == "success"
    assert len(legacy["measurements"]) == 1
    assert legacy["measurements"][0]["name"] == "mean_ndvi"
    assert legacy["measurements"][0]["value"] == 0.55
    assert len(legacy["findings"]) == 1
    assert legacy["findings"][0]["description"] == "Vegetation index computed successfully"


def test_multi_domain_request():
    sub1 = ResearchRequest(
        domain=DomainEnum.EARTH_OBSERVATION,
        task="vegetation_analysis",
        research_question="EO vegetation assessment",
    )
    sub2 = ResearchRequest(
        domain=DomainEnum.CLIMATE,
        task="climate_trends",
        research_question="Temperature warming trends",
    )
    multi = MultiDomainRequest(
        original_question="Correlate regional drought with canopy loss",
        subtasks=[sub1, sub2],
    )
    assert len(multi.subtasks) == 2
    assert multi.execution_mode == "parallel"

