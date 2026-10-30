# -*- coding: utf-8 -*-
"""Health check manager for distributed ICARUS worker nodes."""
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
import httpx

from icarus.core.worker_registry import WorkerRegistry
from icarus.schemas.base import WorkerRuntimeStatus
from icarus.schemas.worker import HealthCheckResponse, WorkerRegistration

logger = logging.getLogger("icarus.health_checker")


class HealthChecker:
    """Manages periodic and on-demand health monitoring of Kaggle worker runtimes."""

    def __init__(
        self,
        registry: WorkerRegistry,
        timeout_seconds: float = 5.0,
        max_consecutive_failures: int = 3,
    ):
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.max_consecutive_failures = max_consecutive_failures

    def check_worker(self, worker_id: str) -> Optional[HealthCheckResponse]:
        """Probes GET /health on a single worker and updates its status."""
        worker = self.registry.get_worker(worker_id)
        if not worker:
            logger.warning("Attempted health check on unregistered worker %s", worker_id)
            return None

        health_url = f"{worker.endpoint_url}/health"
        start_t = time.perf_counter()

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.get(health_url)
                latency = round((time.perf_counter() - start_t) * 1000, 2)

                if resp.status_code == 200:
                    data = resp.json()
                    health = HealthCheckResponse.model_validate(data)

                    worker.runtime_status = health.status
                    worker.gpu_vram_total_mb = health.gpu_vram_total_mb
                    worker.gpu_vram_free_mb = health.gpu_vram_free_mb
                    worker.loaded_models = health.loaded_models
                    worker.last_health_check = datetime.now(timezone.utc)
                    worker.latency_ms = latency
                    worker.consecutive_failures = 0
                    return health
                else:
                    self._record_failure(worker, f"HTTP {resp.status_code}")
                    return None

        except Exception as e:
            self._record_failure(worker, str(e))
            return None

    def check_all_workers(self) -> Dict[str, Optional[HealthCheckResponse]]:
        """Probes all registered workers and returns health summaries."""
        results = {}
        for wid in list(self.registry.workers.keys()):
            results[wid] = self.check_worker(wid)
        return results

    def _record_failure(self, worker: WorkerRegistration, reason: str):
        worker.consecutive_failures += 1
        worker.last_health_check = datetime.now(timezone.utc)
        logger.warning(
            "Health check failed for %s (%s): %s [fail count: %d]",
            worker.worker_id,
            worker.endpoint_url,
            reason,
            worker.consecutive_failures,
        )

        if worker.consecutive_failures >= self.max_consecutive_failures:
            worker.runtime_status = WorkerRuntimeStatus.OFFLINE
            logger.error("Worker %s marked OFFLINE after %d failures", worker.worker_id, worker.consecutive_failures)
        else:
            worker.runtime_status = WorkerRuntimeStatus.DEGRADED

