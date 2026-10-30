# -*- coding: utf-8 -*-
"""Pydantic response schemas for ICARUS distributed execution."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from icarus.schemas.base import ArtifactRef, DomainEnum, TaskStatus


class TaskMetric(BaseModel):
    """Quantitative measurement extracted from scientific execution."""
    name: str
    value: Any
    unit: str
    description: Optional[str] = None
    uncertainty: Optional[float] = None
    precision: Optional[float] = None


class ScientificFinding(BaseModel):
    """High-level qualitative or categorical finding."""
    type: str
    description: str
    value: Optional[Any] = None
    unit: Optional[str] = None
    confidence_score: Optional[float] = None
    interpretation: Optional[str] = None


class ExecutionMetadata(BaseModel):
    """Hardware, timing, and environment metadata for scientific reproducibility."""
    duration_seconds: float
    worker_id: str
    device: str
    vram_used_mb: Optional[int] = None
    host_os: Optional[str] = None
    python_version: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ResearchResponse(BaseModel):
    """Universal structured response returned by a worker after processing a task."""
    model_config = ConfigDict(use_enum_values=True)

    request_id: str
    task_id: str
    worker_id: str
    domain: DomainEnum
    status: TaskStatus
    model_used: Optional[str] = None
    model_version: Optional[str] = None
    dataset_used: Optional[str] = None
    data_source: Optional[Dict[str, Any]] = None
    results: Dict[str, Any] = Field(default_factory=dict, description="Summary, findings, and analysis payloads")
    metrics: List[TaskMetric] = Field(default_factory=list)
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    visualizations: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    uncertainty: Dict[str, Any] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)
    execution_metadata: Optional[ExecutionMetadata] = None
    error: Optional[Dict[str, Any]] = None

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Ensures backward compatibility with the original ICARUS ResearchResult dictionary format."""
        findings = self.results.get("findings", [])
        if not findings and "summary" in self.results:
            findings = [{"type": "summary", "description": self.results["summary"]}]

        measurements = [
            {"name": m.name, "value": m.value, "unit": m.unit}
            for m in self.metrics
        ]

        return {
            "research_question": self.results.get("research_question", ""),
            "domain": self.domain if isinstance(self.domain, str) else self.domain.value,
            "subdomain": self.results.get("subdomain", ""),
            "status": self.status if isinstance(self.status, str) else self.status.value,
            "summary": self.results.get("summary", ""),
            "findings": findings,
            "measurements": measurements,
            "artifacts": [a.model_dump() for a in self.artifacts],
            "visualization_data": self.visualizations,
            "methodology": self.provenance.get("methodology", []),
            "data_sources": [self.data_source] if self.data_source else [],
            "models_used": [self.model_used] if self.model_used else [],
            "validation": self.results.get("validation", {}),
            "uncertainty": self.uncertainty,
            "limitations": self.limitations,
            "provenance": self.provenance,
            "warnings": self.results.get("warnings", [])
        }


class MultiDomainResponse(BaseModel):
    """Synthesized research result spanning multiple workers and domains."""
    request_id: str
    original_question: str
    status: TaskStatus
    subtask_responses: List[ResearchResponse]
    synthesized_summary: str
    overall_provenance: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
