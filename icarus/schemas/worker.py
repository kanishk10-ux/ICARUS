# -*- coding: utf-8 -*-
"""Pydantic schemas for worker registration, discovery, and health management."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from icarus.schemas.base import CapabilityStatus, DomainEnum, WorkerRuntimeStatus


class CapabilityDescriptor(BaseModel):
    """Specific task/capability supported by a worker runtime."""
    id: str
    domain: DomainEnum
    name: str
    status: CapabilityStatus = CapabilityStatus.IMPLEMENTED
    required_models: List[str] = Field(default_factory=list)
    algorithms: List[str] = Field(default_factory=list)
    requires_gpu: bool = False
    notes: Optional[str] = None


class WorkerCapabilitiesResponse(BaseModel):
    """Payload returned by GET /capabilities on a worker node."""
    worker_id: str
    supported_domains: List[DomainEnum]
    capabilities: List[CapabilityDescriptor]
    available_models: List[str]
    loaded_models: List[str]
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_vram_total_mb: int = 0
    gpu_vram_free_mb: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HealthCheckResponse(BaseModel):
    """Payload returned by GET /health on a worker node."""
    worker_id: str
    status: WorkerRuntimeStatus
    uptime_seconds: float
    gpu_available: bool
    gpu_vram_total_mb: int = 0
    gpu_vram_free_mb: int = 0
    loaded_models: List[str] = Field(default_factory=list)
    active_tasks: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class UnloadModelRequest(BaseModel):
    """Request payload for POST /unload-model."""
    model_id: str
    force: bool = False


class UnloadModelResponse(BaseModel):
    """Response payload for POST /unload-model."""
    worker_id: str
    model_id: str
    unloaded: bool
    message: str
    gpu_vram_freed_mb: Optional[int] = None
    current_loaded_models: List[str] = Field(default_factory=list)


class WorkerRegistration(BaseModel):
    """Orchestrator's internal record for an active worker node."""
    worker_id: str
    name: str
    endpoint_url: str
    supported_domains: List[DomainEnum]
    capabilities: Dict[str, CapabilityDescriptor] = Field(default_factory=dict)
    available_models: List[str] = Field(default_factory=list)
    loaded_models: List[str] = Field(default_factory=list)
    gpu_available: bool = False
    gpu_vram_total_mb: int = 0
    gpu_vram_free_mb: int = 0
    runtime_status: WorkerRuntimeStatus = WorkerRuntimeStatus.HEALTHY
    last_health_check: Optional[datetime] = None
    latency_ms: Optional[float] = None
    consecutive_failures: int = 0

