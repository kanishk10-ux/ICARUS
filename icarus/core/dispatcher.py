# -*- coding: utf-8 -*-
"""Task Dispatcher for routing scientific workloads to distributed Kaggle workers."""
import asyncio
import logging
import time
from typing import Dict, List, Optional
import httpx

from icarus.core.worker_registry import WorkerRegistry
from icarus.schemas.base import TaskStatus
from icarus.schemas.request import MultiDomainRequest, ResearchRequest
from icarus.schemas.response import (
    ExecutionMetadata,
    MultiDomainResponse,
    ResearchResponse,
)

logger = logging.getLogger("icarus.dispatcher")


class TaskDispatcher:
    """Dispatches validated scientific tasks to distributed worker runtimes with fault tolerance."""

    def __init__(
        self,
        worker_registry: WorkerRegistry,
        max_retries: int = 2,
        backoff_factor: float = 1.5,
    ):
        self.registry = worker_registry
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def dispatch(self, request: ResearchRequest) -> ResearchResponse:
        """Synchronously dispatches a single task request to an available worker."""
        start_time = time.perf_counter()

        # 1. Discover suitable worker
        candidates = self.registry.find_workers_for_task(
            domain=request.domain,
            task=request.task,
            required_models=request.required_models,
        )

        if not candidates:
            duration = round(time.perf_counter() - start_time, 3)
            logger.warning(
                "No active worker available for domain=%s task=%s",
                request.domain,
                request.task,
            )
            return ResearchResponse(
                request_id=request.request_id,
                task_id=request.task_id,
                worker_id="none",
                domain=request.domain,
                status=TaskStatus.UNAVAILABLE,
                results={"error": f"No active worker registered for domain '{request.domain}' and task '{request.task}'"},
                limitations=["Worker runtime unavailable or offline in current deployment"],
                execution_metadata=ExecutionMetadata(
                    duration_seconds=duration,
                    worker_id="orchestrator",
                    device="cpu",
                ),
                error={
                    "code": "WORKER_UNAVAILABLE",
                    "message": f"No worker found supporting domain '{request.domain}' and task '{request.task}'",
                },
            )

        # Pick the healthiest candidate (e.g. least busy / lowest latency)
        selected_worker = candidates[0]
        analyze_url = f"{selected_worker.endpoint_url}/analyze"

        last_error = None
        for attempt in range(1, self.max_retries + 2):
            try:
                logger.info(
                    "Dispatching task %s to %s (attempt %d/%d)",
                    request.task_id,
                    selected_worker.endpoint_url,
                    attempt,
                    self.max_retries + 1,
                )

                with httpx.Client(timeout=request.timeout_seconds) as client:
                    resp = client.post(analyze_url, json=request.model_dump())

                    if resp.status_code == 200:
                        data = resp.json()
                        response = ResearchResponse.model_validate(data)
                        logger.info("Task %s completed with status=%s", request.task_id, response.status)
                        return response
                    elif resp.status_code in (502, 503, 504):
                        # Transient gateway or service unavailable (e.g. ngrok reconnecting)
                        last_error = f"HTTP {resp.status_code}: {resp.text}"
                        time.sleep(self.backoff_factor ** attempt)
                        continue
                    else:
                        # Permanent worker-side rejection or 4xx
                        data = resp.json() if resp.headers.get("content-type") == "application/json" else {}
                        duration = round(time.perf_counter() - start_time, 3)
                        return ResearchResponse(
                            request_id=request.request_id,
                            task_id=request.task_id,
                            worker_id=selected_worker.worker_id,
                            domain=request.domain,
                            status=TaskStatus.FAILED,
                            results={"error": f"Worker returned HTTP {resp.status_code}"},
                            execution_metadata=ExecutionMetadata(
                                duration_seconds=duration,
                                worker_id=selected_worker.worker_id,
                                device="unknown",
                            ),
                            error={"code": f"HTTP_{resp.status_code}", "detail": data or resp.text},
                        )

            except Exception as e:
                last_error = str(e)
                logger.warning("Attempt %d failed for task %s: %s", attempt, request.task_id, str(e))
                if attempt <= self.max_retries:
                    time.sleep(self.backoff_factor ** attempt)

        # If all retries exhausted:
        duration = round(time.perf_counter() - start_time, 3)
        selected_worker.consecutive_failures += 1
        return ResearchResponse(
            request_id=request.request_id,
            task_id=request.task_id,
            worker_id=selected_worker.worker_id,
            domain=request.domain,
            status=TaskStatus.FAILED,
            results={"error": f"Task dispatch failed after {self.max_retries + 1} attempts"},
            limitations=["Worker connection timed out or network tunnel closed"],
            execution_metadata=ExecutionMetadata(
                duration_seconds=duration,
                worker_id=selected_worker.worker_id,
                device="unknown",
            ),
            error={"code": "DISPATCH_FAILED", "detail": last_error},
        )

    def dispatch_multi(self, multi_request: MultiDomainRequest) -> MultiDomainResponse:
        """Dispatches a set of decomposed subtasks across appropriate domain workers."""
        subtask_responses: List[ResearchResponse] = []
        overall_status = TaskStatus.SUCCESS

        for subtask in multi_request.subtasks:
            resp = self.dispatch(subtask)
            subtask_responses.append(resp)
            if resp.status != TaskStatus.SUCCESS:
                if overall_status == TaskStatus.SUCCESS:
                    overall_status = TaskStatus.PARTIAL

        if all(r.status != TaskStatus.SUCCESS for r in subtask_responses):
            overall_status = TaskStatus.FAILED

        # Build synthesized summary from subtask responses
        summaries = []
        for r in subtask_responses:
            s = r.results.get("summary") or f"Subtask {r.domain} status: {r.status}"
            summaries.append(f"[{r.domain}]: {s}")

        synthesized = "\n".join(summaries)

        return MultiDomainResponse(
            request_id=multi_request.request_id,
            original_question=multi_request.original_question,
            status=overall_status,
            subtask_responses=subtask_responses,
            synthesized_summary=synthesized,
            overall_provenance={
                "subtask_count": len(subtask_responses),
                "successful_subtasks": sum(1 for r in subtask_responses if r.status == TaskStatus.SUCCESS),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

