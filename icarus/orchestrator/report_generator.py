# -*- coding: utf-8 -*-
"""Scientific Research Report Generator powered by GPT-OSS-120B & Deterministic Synthesis."""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import httpx

from icarus.schemas.base import TaskStatus
from icarus.schemas.response import MultiDomainResponse, ResearchResponse

logger = logging.getLogger("icarus.report_generator")


class ReportGenerator:
    """Generates publication-quality scientific reports from distributed worker results."""

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = "openai/gpt-oss-120b"

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model_name = model_name or os.environ.get("GROQ_MODEL", self.DEFAULT_MODEL)

    def generate_report(
        self,
        question: str,
        multi_response: MultiDomainResponse,
        plan_rationale: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesizes worker results into a structured research report and markdown document."""
        # 1. Gather all findings, metrics, artifacts, and limitations
        all_metrics = []
        all_findings = []
        all_artifacts = []
        all_limitations = []
        all_warnings = list(multi_response.warnings)
        data_sources = []
        models_used = []

        for resp in multi_response.subtask_responses:
            for m in resp.metrics:
                all_metrics.append({
                    "domain": resp.domain if isinstance(resp.domain, str) else resp.domain.value,
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "uncertainty": m.uncertainty,
                })
            if "findings" in resp.results and isinstance(resp.results["findings"], list):
                all_findings.extend(resp.results["findings"])
            elif "summary" in resp.results:
                all_findings.append({
                    "type": "summary",
                    "description": resp.results["summary"],
                    "domain": resp.domain if isinstance(resp.domain, str) else resp.domain.value,
                })

            for a in resp.artifacts:
                all_artifacts.append(a.model_dump())

            all_limitations.extend(resp.limitations)
            if resp.data_source:
                data_sources.append(resp.data_source)
            if resp.model_used:
                models_used.append(resp.model_used)

            if resp.status != TaskStatus.SUCCESS:
                all_warnings.append(f"Subtask for domain '{resp.domain}' ended with status: {resp.status}")

        # 2. Try LLM synthesis if available
        markdown_text = None
        executive_summary = multi_response.synthesized_summary

        if self.api_key:
            try:
                llm_summary, llm_md = self._call_groq_synthesis(
                    question=question,
                    subtasks=multi_response.subtask_responses,
                    plan_rationale=plan_rationale,
                )
                if llm_summary:
                    executive_summary = llm_summary
                if llm_md:
                    markdown_text = llm_md
            except Exception as e:
                logger.warning("LLM report synthesis failed: %s; using deterministic template.", str(e))
                all_warnings.append(f"LLM synthesis unavailable ({str(e)}); deterministic report generated.")

        if not markdown_text:
            markdown_text = self._deterministic_markdown_report(
                question=question,
                multi_response=multi_response,
                metrics=all_metrics,
                findings=all_findings,
                limitations=all_limitations,
                warnings=all_warnings,
            )

        report_dict = {
            "research_question": question,
            "overall_status": multi_response.status,
            "executive_summary": executive_summary,
            "markdown_report": markdown_text,
            "findings": all_findings,
            "measurements": all_metrics,
            "artifacts": all_artifacts,
            "data_sources": data_sources,
            "models_used": models_used,
            "limitations": list(set(all_limitations)),
            "warnings": all_warnings,
            "provenance": multi_response.overall_provenance,
        }

        return report_dict

    def _call_groq_synthesis(
        self,
        question: str,
        subtasks: List[ResearchResponse],
        plan_rationale: Optional[str],
    ) -> Tuple[str, str]:
        """Calls GPT-OSS-120B on Groq to generate executive synthesis and scientific markdown."""
        prompt_data = {
            "research_question": question,
            "plan_rationale": plan_rationale,
            "subtask_results": [
                {
                    "domain": s.domain if isinstance(s.domain, str) else s.domain.value,
                    "status": s.status if isinstance(s.status, str) else s.status.value,
                    "model_used": s.model_used,
                    "results": s.results,
                    "metrics": [m.model_dump() for m in s.metrics],
                    "limitations": s.limitations,
                }
                for s in subtasks
            ],
        }

        system_prompt = """You are the scientific reporting engine of ICARUS.
Your role is to synthesize returned multi-domain scientific results into an authoritative research report.
STRICT RULES:
1. Do NOT fabricate numbers, measurements, or confidence scores not present in the results.
2. If any subtask failed or was unavailable, honestly state the failure and its impact.
3. Output MUST be valid JSON with two fields:
   - "executive_summary": A concise paragraph summarizing the key scientific findings.
   - "markdown_report": A comprehensive Markdown report with sections: # Research Overview, ## Methodology & Cross-Domain Decomposition, ## Quantitative Findings & Metrics, ## Scientific Interpretation, ## Limitations & Uncertainties."""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(prompt_data, indent=2)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_tokens": 2048,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(self.GROQ_API_URL, json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Groq API synthesis failed HTTP {resp.status_code}")

            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed.get("executive_summary", ""), parsed.get("markdown_report", "")

    def _deterministic_markdown_report(
        self,
        question: str,
        multi_response: MultiDomainResponse,
        metrics: List[Dict[str, Any]],
        findings: List[Dict[str, Any]],
        limitations: List[str],
        warnings: List[str],
    ) -> str:
        """Constructs a deterministic Markdown report when LLM is offline."""
        lines = [
            f"# ICARUS Scientific Research Report",
            f"**Research Question:** {question}",
            f"**Overall Execution Status:** `{multi_response.status}`",
            f"",
            f"## 1. Executive Summary",
            multi_response.synthesized_summary or "No executive summary available.",
            f"",
            f"## 2. Multi-Domain Findings",
        ]

        if findings:
            for f in findings:
                desc = f.get("description", str(f))
                val = f.get("value")
                unit = f.get("unit", "")
                val_str = f" (`{val} {unit}`)" if val is not None else ""
                lines.append(f"- **{f.get('type', 'Finding')}**: {desc}{val_str}")
        else:
            lines.append("No qualitative findings recorded.")

        lines.extend([
            f"",
            f"## 3. Quantitative Measurements",
        ])

        if metrics:
            lines.append("| Domain | Measurement | Value | Unit | Uncertainty |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for m in metrics:
                unc = m.get("uncertainty") or "N/A"
                lines.append(f"| {m['domain']} | {m['name']} | {m['value']} | {m['unit']} | {unc} |")
        else:
            lines.append("No quantitative measurements recorded.")

        lines.extend([
            f"",
            f"## 4. Subtask Execution Lineage",
        ])

        for st in multi_response.subtask_responses:
            lines.append(f"- **Domain `{st.domain}`** (Task: `{st.task_id[:8]}` on Worker `{st.worker_id}`): Status `{st.status}`")
            if st.model_used:
                lines.append(f"  - Model: `{st.model_used}`")
            if st.limitations:
                lines.append(f"  - Limitations: {', '.join(st.limitations)}")

        if limitations:
            lines.extend([
                f"",
                f"## 5. Scientific Limitations & Boundary Conditions",
            ])
            for lim in set(limitations):
                lines.append(f"- {lim}")

        if warnings:
            lines.extend([
                f"",
                f"## 6. Execution Warnings",
            ])
            for w in warnings:
                lines.append(f"- ⚠️ {w}")

        return "\n".join(lines)
