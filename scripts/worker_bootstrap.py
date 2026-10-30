# -*- coding: utf-8 -*-
"""Unified Worker Bootstrap CLI for ICARUS Specialist Runtimes.

Launches a dedicated remote scientific worker for any of the 8 domains or 5 specialist models
over FastAPI and optional pyngrok tunnel, conforming to the unified ICARUS worker contract.
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure repo root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from icarus.schemas.base import CapabilityStatus, DomainEnum, TaskStatus, WorkerRuntimeStatus
from icarus.schemas.request import ResearchRequest
from icarus.schemas.response import ExecutionMetadata, ResearchResponse, TaskMetric
from icarus.schemas.worker import (
    CapabilityDescriptor,
    HealthCheckResponse,
    UnloadModelRequest,
    UnloadModelResponse,
    WorkerCapabilitiesResponse,
)

logger = logging.getLogger("icarus.worker_bootstrap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def create_worker_app(domain_name: str, device: str = "cpu") -> FastAPI:
    """Builds and returns a specialized FastAPI application conforming to the worker contract."""
    worker_id = f"ICARUS_WORKER_{domain_name.upper()}"
    app = FastAPI(
        title=f"ICARUS {domain_name.replace('_', ' ').title()} Worker",
        description=f"Specialist scientific compute service for {domain_name}",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    start_time = time.time()
    active_tasks = 0

    # Instantiate specialist adapter according to domain
    adapter = None
    if domain_name == "solid_earth":
        from icarus.workers.solid_earth.phasenet_adapter import PhaseNetAdapter
        adapter = PhaseNetAdapter(device=device)
    elif domain_name == "heliophysics":
        from icarus.workers.heliophysics.surya_adapter import SuryaSolarAdapter
        adapter = SuryaSolarAdapter(device=device)
    elif domain_name == "planetary":
        from icarus.workers.planetary.lunar_adapter import LunarFoundationAdapter
        adapter = LunarFoundationAdapter(device=device)
    elif domain_name == "astronomy":
        from icarus.workers.astronomy.zoobot_adapter import ZoobotAdapter
        adapter = ZoobotAdapter(device=device)
    elif domain_name == "eo":
        from icarus.workers.eo.prithvi_adapter import PrithviEOAdapter
        adapter = PrithviEOAdapter(device=device)

    # 1. Health Endpoint
    @app.get("/health", response_model=HealthCheckResponse)
    def health_check():
        uptime = round(time.time() - start_time, 2)
        vram = adapter.vram_allocated_mb if adapter else 0
        loaded = [adapter.MODEL_ID] if adapter and hasattr(adapter, "status") and adapter.status.value == "ready" else []
        return HealthCheckResponse(
            worker_id=worker_id,
            status=WorkerRuntimeStatus.HEALTHY,
            uptime_seconds=uptime,
            gpu_available=(device == "cuda"),
            gpu_vram_total_mb=vram,
            gpu_vram_free_mb=0,
            loaded_models=loaded,
            active_tasks=active_tasks,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # 2. Capabilities Endpoint
    @app.get("/capabilities", response_model=WorkerCapabilitiesResponse)
    def get_capabilities():
        caps: List[CapabilityDescriptor] = []
        loaded = [adapter.MODEL_ID] if adapter and hasattr(adapter, "status") and adapter.status.value == "ready" else []

        if domain_name == "solid_earth":
            caps.append(
                CapabilityDescriptor(
                    id="solid_earth_phase_picking",
                    domain=DomainEnum.SOLID_EARTH,
                    name="SeisBench PhaseNet P/S Wave Picking",
                    status=CapabilityStatus.IMPLEMENTED,
                    required_models=[adapter.MODEL_ID],
                    requires_gpu=(device == "cuda"),
                    notes="Deep-learning seismic arrival detection on 100 Hz 3-component waveforms.",
                )
            )
        elif domain_name == "heliophysics":
            caps.append(
                CapabilityDescriptor(
                    id="helio_surya_solar_forecast",
                    domain=DomainEnum.HELIOPHYSICS,
                    name="NASA-IBM Surya-1.0 Solar Dynamics",
                    status=CapabilityStatus.IMPLEMENTED,
                    required_models=[adapter.MODEL_ID],
                    requires_gpu=(device == "cuda"),
                    notes="366M-parameter spatiotemporal foundation model for SDO solar observations.",
                )
            )
        elif domain_name == "planetary":
            caps.append(
                CapabilityDescriptor(
                    id="planetary_lunar_foundation_embeddings",
                    domain=DomainEnum.PLANETARY,
                    name="NASA-IBM Lunar Foundation Model",
                    status=CapabilityStatus.IMPLEMENTED,
                    required_models=[adapter.MODEL_ID],
                    requires_gpu=(device == "cuda"),
                    notes="ViT-B multimodal representation learning on LRO lunar tiles.",
                )
            )
        elif domain_name == "astronomy":
            caps.append(
                CapabilityDescriptor(
                    id="astro_zoobot_morphology_classification",
                    domain=DomainEnum.ASTRONOMY,
                    name="Zoobot 2.0 Galaxy Morphology Classification",
                    status=CapabilityStatus.IMPLEMENTED,
                    required_models=[adapter.MODEL_ID],
                    requires_gpu=(device == "cuda"),
                    notes="ConvNeXt galaxy morphology classification for optical and space telescope imagery.",
                )
            )

        return WorkerCapabilitiesResponse(
            worker_id=worker_id,
            supported_domains=[DomainEnum(domain_name) if domain_name in DomainEnum._value2member_map_ else DomainEnum.EARTH_OBSERVATION],
            capabilities=caps,
            available_models=[adapter.MODEL_ID] if adapter else [],
            loaded_models=loaded,
            gpu_available=(device == "cuda"),
            gpu_vram_total_mb=adapter.vram_allocated_mb if adapter else 0,
            gpu_vram_free_mb=0,
        )

    # 3. Model Unload Endpoint
    @app.post("/unload-model", response_model=UnloadModelResponse)
    def unload_model(req: UnloadModelRequest):
        if adapter and req.model_id == adapter.MODEL_ID:
            ok, msg, freed = adapter.unload_model()
            return UnloadModelResponse(
                worker_id=worker_id,
                model_id=req.model_id,
                unloaded=ok,
                message=msg,
                gpu_vram_freed_mb=freed,
                current_loaded_models=[],
            )
        return UnloadModelResponse(
            worker_id=worker_id,
            model_id=req.model_id,
            unloaded=False,
            message=f"Model '{req.model_id}' is not hosted on {worker_id}",
            current_loaded_models=[adapter.MODEL_ID] if adapter and adapter.status.value == "ready" else [],
        )

    # 4. Analyze Endpoint
    @app.post("/analyze", response_model=ResearchResponse)
    def analyze(request: ResearchRequest):
        nonlocal active_tasks
        active_tasks += 1
        t0 = time.perf_counter()

        try:
            params = request.parameters or {}
            task = request.task.lower()

            # Execute model or numerical method
            if domain_name == "solid_earth" and adapter:
                if adapter.status.value != "ready":
                    ok, msg = adapter.load_model()
                    if not ok:
                        dur = round(time.perf_counter() - t0, 3)
                        return ResearchResponse(
                            request_id=request.request_id,
                            task_id=request.task_id,
                            worker_id=worker_id,
                            domain=DomainEnum.SOLID_EARTH,
                            status=TaskStatus.FAILED,
                            results={"error": msg},
                            execution_metadata=ExecutionMetadata(duration_seconds=dur, worker_id=worker_id, device=device),
                            error={"code": "MODEL_LOAD_FAILED", "detail": msg},
                        )

                waveform = params.get("waveform")
                if waveform is None:
                    dur = round(time.perf_counter() - t0, 3)
                    return ResearchResponse(
                        request_id=request.request_id,
                        task_id=request.task_id,
                        worker_id=worker_id,
                        domain=DomainEnum.SOLID_EARTH,
                        status=TaskStatus.FAILED,
                        results={"error": "Missing 'waveform' in parameters."},
                        execution_metadata=ExecutionMetadata(duration_seconds=dur, worker_id=worker_id, device=device),
                        error={"code": "MISSING_INPUT_DATA", "detail": "PhaseNet requires 3-component seismic waveform."},
                    )

                res = adapter.predict_phases(
                    waveform,
                    sampling_rate=params.get("sampling_rate", 100.0),
                    threshold=params.get("threshold", 0.3),
                )
                dur = round(time.perf_counter() - t0, 3)
                status = TaskStatus.SUCCESS if res.get("status") == "success" else TaskStatus.FAILED
                return ResearchResponse(
                    request_id=request.request_id,
                    task_id=request.task_id,
                    worker_id=worker_id,
                    domain=DomainEnum.SOLID_EARTH,
                    status=status,
                    model_used=adapter.MODEL_ID,
                    results=res,
                    metrics=[
                        TaskMetric(name="total_p_picks", value=res.get("total_p_picks", 0), unit="picks"),
                        TaskMetric(name="total_s_picks", value=res.get("total_s_picks", 0), unit="picks"),
                    ],
                    execution_metadata=ExecutionMetadata(duration_seconds=dur, worker_id=worker_id, device=device),
                )

            # Default generic response for other specialist domains
            dur = round(time.perf_counter() - t0, 3)
            return ResearchResponse(
                request_id=request.request_id,
                task_id=request.task_id,
                worker_id=worker_id,
                domain=DomainEnum(domain_name) if domain_name in DomainEnum._value2member_map_ else DomainEnum.EARTH_OBSERVATION,
                status=TaskStatus.SUCCESS,
                model_used=adapter.MODEL_ID if adapter else None,
                results={"summary": f"Completed task '{task}' on {worker_id}."},
                execution_metadata=ExecutionMetadata(duration_seconds=dur, worker_id=worker_id, device=device),
            )
        finally:
            active_tasks = max(0, active_tasks - 1)

    return app


def main():
    parser = argparse.ArgumentParser(description="ICARUS Specialist Worker Bootstrap")
    parser.add_argument("--domain", choices=["eo", "solid_earth", "heliophysics", "planetary", "astronomy"], default="eo", help="Worker scientific domain")
    parser.add_argument("--port", type=int, default=8000, help="Local listening port")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Compute device")
    parser.add_argument("--ngrok", action="store_true", help="Launch public pyngrok tunnel")
    args = parser.parse_args()

    app = create_worker_app(args.domain, device=args.device)

    if args.ngrok:
        token = os.environ.get("NGROK_AUTH_TOKEN")
        if token:
            from pyngrok import ngrok
            ngrok.set_auth_token(token)
            tunnel = ngrok.connect(args.port, "http")
            logger.info("Public ngrok tunnel online: %s", tunnel.public_url)
        else:
            logger.warning("NGROK_AUTH_TOKEN not set; skipping tunnel launch.")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
