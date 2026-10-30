# -*- coding: utf-8 -*-
"""Pydantic request schemas for ICARUS distributed execution."""
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from icarus.schemas.base import ArtifactRef, DomainEnum, SpatialBounds, TemporalRange


class ResearchRequest(BaseModel):
    """Universal task request dispatched from Orchestrator to a Worker."""
    model_config = ConfigDict(use_enum_values=True)

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    domain: DomainEnum = Field(..., description="Target scientific domain")
    task: str = Field(..., description="Domain capability/task identifier")
    research_question: str = Field(..., description="Natural language scientific question")
    dataset: Optional[str] = Field(None, description="Specific dataset identifier requested")
    spatial_bounds: Optional[SpatialBounds] = None
    temporal_range: Optional[TemporalRange] = None
    input_artifacts: List[ArtifactRef] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_models: List[str] = Field(default_factory=list)
    timeout_seconds: float = Field(120.0, ge=5.0, le=900.0)
    allow_synthetic_demo: bool = Field(False, description="Whether synthetic fallback is explicitly permitted")


class MultiDomainRequest(BaseModel):
    """Orchestrator-level multi-domain decomposed request."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_question: str = Field(..., description="High-level research question spanning multiple disciplines")
    subtasks: List[ResearchRequest] = Field(..., min_length=1)
    execution_mode: str = Field("parallel", description="'parallel' or 'sequential'")
    parameters: Dict[str, Any] = Field(default_factory=dict)
