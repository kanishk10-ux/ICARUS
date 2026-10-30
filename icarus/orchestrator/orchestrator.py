# -*- coding: utf-8 -*-
"""Central Research Orchestrator for ICARUS distributed 4-account architecture."""
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from icarus.core.dispatcher import TaskDispatcher
from icarus.core.health_checker import HealthChecker
from icarus.core.worker_registry import WorkerRegistry
from icarus.orchestrator.llm_planner import LLMPlanner, StructuredResearchPlan
from icarus.orchestrator.report_generator import ReportGenerator
from icarus.schemas.base import TaskStatus
from icarus.schemas.request import MultiDomainRequest, ResearchRequest
from icarus.schemas.response import MultiDomainResponse, ResearchResponse
from icarus.schemas.worker import UnloadModelResponse

logger = logging.getLogger("icarus.central_orchestrator")


class CentralResearchOrchestrator:
    """The central coordinator (Account 1: ICARUS_CORE) managing planning, dispatch, and reporting."""

    def __init__(
        self,
        worker_registry: Optional[WorkerRegistry] = None,
        llm_planner: Optional[LLMPlanner] = None,
        report_generator: Optional[ReportGenerator] = None,
        task_dispatcher: Optional[TaskDispatcher] = None,
        health_checker: Optional[HealthChecker] = None,
        capability_registry: Optional[Any] = None,
        model_registry: Optional[Any] = None,
        dataset_registry: Optional[Any] = None,
        groq_api_key: Optional[str] = None,
    ):
        self.worker_registry = worker_registry or WorkerRegistry()
        self.health_checker = health_checker or HealthChecker(self.worker_registry)
        self.task_dispatcher = task_dispatcher or TaskDispatcher(self.worker_registry)

        self.capability_registry = capability_registry
        self.model_registry = model_registry
        self.dataset_registry = dataset_registry

        self.llm_planner = llm_planner or LLMPlanner(
            api_key=groq_api_key,
            capability_registry=self.capability_registry,
            worker_registry=self.worker_registry,
        )

        self.report_generator = report_generator or ReportGenerator(
            api_key=groq_api_key
        )

    def register_worker(self, endpoint_url: str, name: Optional[str] = None) -> Tuple[bool, str]:
        """Registers a remote worker node via its public ngrok or HTTP endpoint."""
        success, msg, _ = self.worker_registry.register_worker_endpoint(endpoint_url, name=name)
        return success, msg

    def check_workers_health(self) -> Dict[str, Any]:
        """Probes all registered workers and returns health statuses and latency."""
        return self.health_checker.check_all_workers()

    def unload_model(self, worker_id: str, model_id: str, force: bool = False) -> UnloadModelResponse:
        """Remotely requests a worker to unload a model from GPU VRAM."""
        return self.worker_registry.unload_model_on_worker(worker_id, model_id, force=force)

    def execute_research(
        self,
        question: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], MultiDomainResponse, StructuredResearchPlan]:
        """Main end-to-end scientific research execution pipeline:

        1. Health check & worker discovery
        2. LLM / Groq research planning & domain decomposition
        3. Capability validation
        4. Task dispatching to remote workers (with fault-tolerant retries)
        5. Scientific synthesis and publication-quality report generation
        """
        start_t = time.perf_counter()
        params = parameters or {}

        # 1. Probe registered workers to refresh statuses
        self.health_checker.check_all_workers()

        # 2. Generate structured plan via GPT-OSS-120B (with deterministic fallback)
        multi_req, plan, plan_warnings = self.llm_planner.plan_research(
            question=question, user_parameters=params
        )

        # 3. Dispatch tasks to appropriate distributed workers
        multi_response = self.task_dispatcher.dispatch_multi(multi_req)
        multi_response.warnings.extend(plan_warnings)

        # 4. Synthesize final scientific report
        elapsed_sec = round(time.perf_counter() - start_t, 3)
        multi_response.overall_provenance["total_orchestration_seconds"] = elapsed_sec
        multi_response.overall_provenance["plan"] = plan.model_dump()

        report = self.report_generator.generate_report(
            question=question,
            multi_response=multi_response,
            plan_rationale=plan.scientific_rationale,
        )

        return report, multi_response, plan

