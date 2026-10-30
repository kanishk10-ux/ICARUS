# -*- coding: utf-8 -*-
"""Worker Registry for dynamic discovery and capability routing in ICARUS."""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import httpx

from icarus.schemas.base import DomainEnum, WorkerRuntimeStatus
from icarus.schemas.worker import (
    CapabilityDescriptor,
    UnloadModelRequest,
    UnloadModelResponse,
    WorkerCapabilitiesResponse,
    WorkerRegistration,
)

logger = logging.getLogger("icarus.worker_registry")


class WorkerRegistry:
    """Central registry tracking active, distributed Kaggle worker nodes."""

    def __init__(self, timeout_seconds: float = 10.0):
        self.workers: Dict[str, WorkerRegistration] = {}
        self.timeout_seconds = timeout_seconds

    def register_worker_endpoint(
        self, endpoint_url: str, name: Optional[str] = None
    ) -> Tuple[bool, str, Optional[WorkerRegistration]]:
        """Probes a newly available worker URL (e.g. from ngrok) and registers its capabilities."""
        endpoint = endpoint_url.rstrip("/")
        cap_url = f"{endpoint}/capabilities"

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.get(cap_url)
                if resp.status_code != 200:
                    msg = f"Worker at {endpoint} returned HTTP {resp.status_code} on /capabilities"
                    logger.warning(msg)
                    return False, msg, None

                data = resp.json()
                cap_resp = WorkerCapabilitiesResponse.model_validate(data)

            wid = cap_resp.worker_id
            caps_dict = {c.id: c for c in cap_resp.capabilities}

            registration = WorkerRegistration(
                worker_id=wid,
                name=name or f"Worker-{wid[:8]}",
                endpoint_url=endpoint,
                supported_domains=cap_resp.supported_domains,
                capabilities=caps_dict,
                available_models=cap_resp.available_models,
                loaded_models=cap_resp.loaded_models,
                gpu_available=cap_resp.gpu_available,
                gpu_vram_total_mb=cap_resp.gpu_vram_total_mb,
                gpu_vram_free_mb=cap_resp.gpu_vram_free_mb,
                runtime_status=WorkerRuntimeStatus.HEALTHY,
                last_health_check=datetime.now(timezone.utc),
            )

            self.workers[wid] = registration
            logger.info("Successfully registered worker %s (%s) with %d capabilities", wid, endpoint, len(caps_dict))
            return True, f"Registered worker {wid} with {len(caps_dict)} capabilities", registration

        except Exception as e:
            msg = f"Failed to connect to worker at {endpoint}: {str(e)}"
            logger.error(msg)
            return False, msg, None

    def unregister_worker(self, worker_id: str) -> bool:
        """Removes a worker from active registry."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            logger.info("Deregistered worker %s", worker_id)
            return True
        return False

    def get_worker(self, worker_id: str) -> Optional[WorkerRegistration]:
        return self.workers.get(worker_id)

    def list_workers(self, healthy_only: bool = False) -> List[WorkerRegistration]:
        if healthy_only:
            return [w for w in self.workers.values() if w.runtime_status == WorkerRuntimeStatus.HEALTHY]
        return list(self.workers.values())

    def find_workers_for_task(
        self, domain: DomainEnum, task: str, required_models: Optional[List[str]] = None
    ) -> List[WorkerRegistration]:
        """Finds candidate workers matching the specified domain and capability."""
        candidates = []
        domain_val = domain if isinstance(domain, str) else domain.value

        for w in self.workers.values():
            if w.runtime_status == WorkerRuntimeStatus.OFFLINE:
                continue

            # Check domain support
            worker_domains = [d if isinstance(d, str) else d.value for d in w.supported_domains]
            if domain_val not in worker_domains:
                continue

            # Check task capability
            cap_match = task in w.capabilities or any(
                task.lower() in cid.lower() for cid in w.capabilities
            )
            if not cap_match and w.capabilities:
                continue

            # Check model availability if specific models are requested
            if required_models:
                models_supported = all(
                    m in w.available_models or m in w.loaded_models for m in required_models
                )
                if not models_supported:
                    continue

            candidates.append(w)

        return candidates

    def unload_model_on_worker(
        self, worker_id: str, model_id: str, force: bool = False
    ) -> UnloadModelResponse:
        """Instructs a specific worker to unload a model and reclaim GPU memory."""
        worker = self.get_worker(worker_id)
        if not worker:
            return UnloadModelResponse(
                worker_id=worker_id,
                model_id=model_id,
                unloaded=False,
                message=f"Worker {worker_id} not found in registry",
            )

        unload_url = f"{worker.endpoint_url}/unload-model"
        req = UnloadModelRequest(model_id=model_id, force=force)

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(unload_url, json=req.model_dump())
                if resp.status_code == 200:
                    data = resp.json()
                    unload_resp = UnloadModelResponse.model_validate(data)
                    worker.loaded_models = unload_resp.current_loaded_models
                    return unload_resp
                return UnloadModelResponse(
                    worker_id=worker_id,
                    model_id=model_id,
                    unloaded=False,
                    message=f"Worker returned HTTP {resp.status_code}",
                )
        except Exception as e:
            return UnloadModelResponse(
                worker_id=worker_id,
                model_id=model_id,
                unloaded=False,
                message=f"Network error unloading model: {str(e)}",
            )

