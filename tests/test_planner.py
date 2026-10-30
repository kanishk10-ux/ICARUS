"""
Comprehensive Test Suite for Deterministic Multi-Domain Task Router
====================================================================
Tests that ICARUS's deterministic router correctly routes scientific questions
across all 8 domains and 5 specialist models without defaulting to EO vegetation analysis.
"""

import pytest

from icarus.orchestrator.llm_planner import LLMPlanner
from icarus.schemas.base import DomainEnum


@pytest.fixture
def planner():
    # Null API key forces deterministic fallback router
    return LLMPlanner(api_key=None)


def test_routing_solid_earth_phasenet(planner):
    q = "Detect seismic P-wave and S-wave arrival phases for the regional earthquake waveform at 100 Hz"
    multi_req, plan, warnings = planner.plan_research(q)

    assert plan.subtasks[0].domain == DomainEnum.SOLID_EARTH
    assert "phasenet" in plan.subtasks[0].required_models[0].lower()
    assert plan.subtasks[0].task == "solid_earth_phase_picking"


def test_routing_heliophysics_surya(planner):
    q = "Analyze SDO AIA solar coronal extreme ultraviolet emissions for M-class solar flare forecasting"
    multi_req, plan, warnings = planner.plan_research(q)

    assert plan.subtasks[0].domain == DomainEnum.HELIOPHYSICS
    assert "surya" in plan.subtasks[0].required_models[0].lower()
    assert plan.subtasks[0].task == "helio_surya_solar_forecast"


def test_routing_planetary_lunar(planner):
    q = "Analyze lunar crater topography, depth to diameter ratio, and crater morphometry from LROC NAC imagery"
    multi_req, plan, warnings = planner.plan_research(q)

    assert plan.subtasks[0].domain == DomainEnum.PLANETARY
    assert "lunar" in plan.subtasks[0].required_models[0].lower()
    assert plan.subtasks[0].task == "planetary_lunar_foundation_embeddings"


def test_routing_astronomy_zoobot(planner):
    q = "Classify galaxy visual morphology (smooth vs disk vs spiral arms) from SDSS DR18 cutout image"
    multi_req, plan, warnings = planner.plan_research(q)

    assert plan.subtasks[0].domain == DomainEnum.ASTRONOMY
    assert "zoobot" in plan.subtasks[0].required_models[0].lower()
    assert plan.subtasks[0].task == "astro_zoobot_morphology_classification"


def test_routing_earth_observation_prithvi(planner):
    q = "Extract foundation spatial representations from multi-temporal Sentinel-2 6-band imagery using Prithvi-EO-2.0"
    multi_req, plan, warnings = planner.plan_research(q)

    assert plan.subtasks[0].domain == DomainEnum.EARTH_OBSERVATION
    assert "prithvi" in plan.subtasks[0].required_models[0].lower()


def test_routing_oceans(planner):
    q = "Detect marine heatwaves and compute sea surface temperature SST anomalies from daily NOAA OISST data"
    multi_req, plan, warnings = planner.plan_research(q)

    assert plan.subtasks[0].domain == DomainEnum.OCEANS
    assert "vegetation" not in plan.subtasks[0].task


def test_routing_atmosphere(planner):
    q = "Compute wind vectors and meteorological wind speed from u and v wind velocity fields"
    multi_req, plan, warnings = planner.plan_research(q)

    assert plan.subtasks[0].domain == DomainEnum.ATMOSPHERE
    assert "vegetation" not in plan.subtasks[0].task


def test_routing_climate(planner):
    q = "Estimate long-term decadal temperature trend slope and Mann-Kendall monotonic test from 40-year observations"
    multi_req, plan, warnings = planner.plan_research(q)

    assert plan.subtasks[0].domain == DomainEnum.CLIMATE
    assert "vegetation" not in plan.subtasks[0].task


def test_no_universal_default_to_vegetation(planner):
    questions_and_expected = [
        ("Seismic ground motion PGA", DomainEnum.SOLID_EARTH),
        ("Coronal mass ejection solar storm", DomainEnum.HELIOPHYSICS),
        ("Lunar terrain slope and aspect from DEM", DomainEnum.PLANETARY),
        ("Lomb-Scargle periodogram stellar light curve", DomainEnum.ASTRONOMY),
        ("Ocean chlorophyll and salinity", DomainEnum.OCEANS),
        ("Atmospheric precipitation accumulation", DomainEnum.ATMOSPHERE),
        ("Drought Standardized Precipitation Index SPI", DomainEnum.CLIMATE),
    ]

    for question, expected_domain in questions_and_expected:
        _, plan, _ = planner.plan_research(question)
        assert plan.subtasks[0].domain == expected_domain, (
            f"Question '{question}' was incorrectly routed to {plan.subtasks[0].domain} instead of {expected_domain}"
        )
