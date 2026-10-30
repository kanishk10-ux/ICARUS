# -*- coding: utf-8 -*-
"""FastAPI Application for ICARUS_CORE (Account 1 Central Orchestrator)."""
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from icarus.core.worker_registry import WorkerRegistry
from icarus.core.health_checker import HealthChecker
from icarus.core.dispatcher import TaskDispatcher
from icarus.orchestrator.orchestrator import CentralResearchOrchestrator
from icarus.schemas.base import WorkerRuntimeStatus
from icarus.schemas.worker import WorkerRegistration

logger = logging.getLogger("icarus.core.service")

app = FastAPI(
    title="ICARUS_CORE Central Orchestrator Service",
    description="Central coordination API for distributed multi-domain scientific research (Account 1)",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core State
START_TIME = time.time()
worker_registry = WorkerRegistry(timeout_seconds=10.0)
health_checker = HealthChecker(worker_registry, timeout_seconds=5.0)
task_dispatcher = TaskDispatcher(worker_registry)

orchestrator = CentralResearchOrchestrator(
    worker_registry=worker_registry,
    health_checker=health_checker,
    task_dispatcher=task_dispatcher,
)


class RegisterWorkerRequest(BaseModel):
    """Payload sent by a distributed worker node on startup to register its dynamic endpoint."""
    endpoint_url: str = Field(..., description="Public ngrok or HTTP endpoint URL")
    worker_name: Optional[str] = Field(None, description="Descriptive worker name (e.g. ICARUS_WORKER_EO)")


class ResearchSubmissionRequest(BaseModel):
    """Payload to initiate an end-to-end distributed research workflow."""
    question: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def core_health() -> Dict[str, Any]:
    """Probes orchestrator health, uptime, and active worker count."""
    uptime = round(time.time() - START_TIME, 2)
    active_count = len([w for w in worker_registry.workers.values() if w.runtime_status == WorkerRuntimeStatus.HEALTHY])
    return {
        "status": "healthy",
        "service": "ICARUS_CORE",
        "uptime_seconds": uptime,
        "registered_workers": len(worker_registry.workers),
        "active_healthy_workers": active_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/register-worker")
def register_worker(req: RegisterWorkerRequest) -> Dict[str, Any]:
    """Dynamically discovers and registers a remote worker node from its announced endpoint."""
    success, msg, reg = worker_registry.register_worker_endpoint(req.endpoint_url, name=req.worker_name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to discover and register worker at {req.endpoint_url}: {msg}",
        )
    return {
        "status": "registered",
        "worker_id": reg.worker_id,
        "worker_name": reg.name,
        "endpoint_url": reg.endpoint_url,
        "supported_domains": reg.supported_domains,
        "capabilities_count": len(reg.capabilities),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/workers")
def list_workers() -> List[Dict[str, Any]]:
    """Lists all registered distributed worker nodes and their current health."""
    health_checker.check_all_workers()
    result = []
    for wid, w in worker_registry.workers.items():
        result.append({
            "worker_id": wid,
            "name": w.name,
            "endpoint_url": w.endpoint_url,
            "status": w.runtime_status,
            "latency_ms": w.latency_ms,
            "supported_domains": w.supported_domains,
            "capabilities": list(w.capabilities.keys()),
            "loaded_models": w.loaded_models,
        })
    return result


@app.post("/research")
def execute_research(req: ResearchSubmissionRequest) -> Dict[str, Any]:
    """Executes a full multi-domain research workflow across registered workers."""
    report, multi_resp, plan = orchestrator.execute_research(req.question, parameters=req.parameters)
    return {
        "report": report,
        "multi_response": multi_resp.model_dump(),
        "plan": plan.model_dump(),
    }

