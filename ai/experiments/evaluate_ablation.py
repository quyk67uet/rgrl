"""Real ablation evaluation for neuro-symbolic architecture variants.

This script compares three real runtime configurations:
1. gnn_only
2. single_llm
3. proposed_3agent

"""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Any, Literal

from ai.dual_system.deliberative_llm import run_deliberative_pipeline
from ai.dual_system.neuro_symbolic_router import orchestrate_workflow_step
from ai.experiments.evaluate_tradeoff import load_test_traces

AblationMode = Literal["gnn_only", "single_llm", "proposed_3agent"]

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CSV_PATH = REPO_ROOT / "ai" / "results" / "real_ablation_results.csv"
EXPECTED_TEST_SIZE = 200

_ALLOWED_SKELETONS: set[tuple[str, ...]] = {
    ("TriageAgent", "EHRAgent", "SummaryAgent"),
    ("TriageAgent", "DispensationAgent", "SummaryAgent"),
    (
        "TriageAgent",
        "EHRAgent",
        "DispensationAgent",
        "EHRAgent",
        "ReconciliationAgent",
        "SummaryAgent",
    ),
}


def _extract_selected_pathway_from_action(action: Any) -> int | None:
    """Extract selected pathway from action payload if available."""
    if not isinstance(action, dict):
        return None

    pathway = action.get("selected_pathway")
    if isinstance(pathway, int):
        return pathway
    if isinstance(pathway, str) and pathway.strip().isdigit():
        return int(pathway.strip())
    return None


def _extract_agent_sequence_from_action(action: Any) -> list[str]:
    """Extract ordered agent sequence from model action steps."""
    if not isinstance(action, dict):
        return []

    steps = action.get("steps")
    if not isinstance(steps, list):
        return []

    sequence: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        agent_name = step.get("agent_name")
        if isinstance(agent_name, str) and agent_name:
            sequence.append(agent_name)

    return sequence


def _is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_terms = (
        "rate limit",
        "too many requests",
        "429",
        "timeout",
        "timed out",
        "temporarily",
        "temporary",
        "unavailable",
        "connection",
        "network",
        "server error",
        "502",
        "503",
        "504",
    )
    return any(term in message for term in retry_terms)


async def _run_mode_on_trace(mode: AblationMode, trace: dict[str, Any]) -> dict[str, Any]:
    """Run one mode on one trace and return normalized execution payload."""
    input_scenario = trace.get("input_scenario")
    if isinstance(input_scenario, dict) and "nl_command" in input_scenario:
        input_context = str(input_scenario.get("nl_command", ""))
    elif "patient_context" in trace:
        input_context = str(trace.get("patient_context", ""))
    else:
        input_context = str(trace.get("context", ""))

    history = trace.get("spans", [])
    if not isinstance(history, list):
        history = trace.get("graph_history", [])
    if not isinstance(history, list):
        history = []

    max_retries = 3
    base_delay = 0.5

    for attempt in range(max_retries + 1):
        try:
            if mode == "gnn_only":
                router_result = await orchestrate_workflow_step(
                    patient_context=input_context,
                    graph_history=history,
                    tau_threshold=0.0,
                )
                return {
                    "selected_system": router_result.get("selected_system"),
                    "action": router_result.get("action"),
                    "raw_result": router_result,
                }

            if mode == "single_llm":
                llm_result = await run_deliberative_pipeline(
                    patient_context=input_context,
                    max_retries=0,
                )
                return {
                    "selected_system": "system2",
                    "action": llm_result.get("action"),
                    "raw_result": llm_result,
                }

            llm_result = await run_deliberative_pipeline(
                patient_context=input_context,
                max_retries=3,
            )
            return {
                "selected_system": "system2",
                "action": llm_result.get("action"),
                "raw_result": llm_result,
            }
        except Exception as exc:  # pragma: no cover - runtime/network dependent.
            if attempt < max_retries and _is_retryable_error(exc):
                delay = base_delay * (2**attempt)
                await asyncio.sleep(delay)
                continue
            return {
                "selected_system": "hitl",
                "action": None,
                "raw_result": {
                    "status": "hitl_required",
                    "selected_system": "system2",
                    "escalate_to_hitl": True,
                    "attempts": attempt + 1,
                    "reason": f"ablation_runtime_error: {exc}",
                },
            }


async def run_ablation_experiment(
    test_cases: list[dict[str, Any]],
    mode: AblationMode,
) -> dict[str, float | int | str]:
    """Run one ablation configuration on real traces.

    Metrics:
    - compliance_rate
    - hallucination_rate
    - redundancy_rate
    - average_steps
    """
    if mode not in {"gnn_only", "single_llm", "proposed_3agent"}:
        raise ValueError("Invalid mode. Use: gnn_only, single_llm, proposed_3agent.")

    if not test_cases:
        raise ValueError("test_cases must contain at least one trace.")

    compliance_hits = 0
    hallucination_hits = 0
    total_steps = 0
    total_redundancy = 0.0

    total = len(test_cases)
    semaphore = asyncio.Semaphore(5)

    async def _run_with_semaphore(index: int, trace: dict[str, Any]) -> dict[str, Any]:
        print(
            f"[ablation:{mode}] Evaluating workflow context {index}/{total}...",
            flush=True,
        )
        async with semaphore:
            return await _run_mode_on_trace(mode=mode, trace=trace)

    tasks = [
        asyncio.create_task(_run_with_semaphore(index, trace))
        for index, trace in enumerate(test_cases, start=1)
    ]
    executions = await asyncio.gather(*tasks, return_exceptions=True)

    for trace, execution in zip(test_cases, executions):
        if isinstance(execution, Exception):  # pragma: no cover - runtime/network dependent.
            print(
                f"[ablation:{mode}] trace={trace.get('trace_id')} failed with error: {execution}",
                flush=True,
            )
            hallucination_hits += 1
            continue

        action = execution.get("action")
        predicted_pathway = _extract_selected_pathway_from_action(action)
        predicted_sequence = _extract_agent_sequence_from_action(action)

        ground_truth_pathway = trace.get("selected_pathway")
        ground_truth_sequence = trace.get("ground_truth_agent_sequence", [])
        if not isinstance(ground_truth_sequence, list):
            ground_truth_sequence = []

        is_compliant = (
            isinstance(ground_truth_pathway, int)
            and predicted_pathway == ground_truth_pathway
        ) or (predicted_sequence == ground_truth_sequence and len(predicted_sequence) > 0)

        if is_compliant:
            compliance_hits += 1

        is_hallucination = tuple(predicted_sequence) not in _ALLOWED_SKELETONS
        if is_hallucination:
            hallucination_hits += 1

        total_steps += len(predicted_sequence)
        gt_steps = max(len(ground_truth_sequence), 1)
        extra_steps = max(len(predicted_sequence) - len(ground_truth_sequence), 0)
        total_redundancy += extra_steps / float(gt_steps)

    count = float(total)
    return {
        "mode": mode,
        "n_cases": total,
        "compliance_rate": compliance_hits / count,
        "hallucination_rate": hallucination_hits / count,
        "redundancy_rate": total_redundancy / count,
        "average_steps": total_steps / count,
    }


async def run_all_modes(
    test_cases: list[dict[str, Any]],
) -> list[dict[str, float | int | str]]:
    """Evaluate all ablation modes sequentially on the same traces."""
    return [
        await run_ablation_experiment(test_cases, "gnn_only"),
        await run_ablation_experiment(test_cases, "single_llm"),
        await run_ablation_experiment(test_cases, "proposed_3agent"),
    ]


def save_ablation_results(rows: list[dict[str, float | int | str]], output_path: Path) -> Path:
    """Persist ablation metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "n_cases",
                "compliance_rate",
                "hallucination_rate",
                "redundancy_rate",
                "average_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> None:
    """CLI entrypoint for real ablation experiment on the test split."""
    test_traces = load_test_traces()
    print(
        "[ablation] Loaded workflow context split with {count} traces (target: {target}).".format(
            count=len(test_traces),
            target=EXPECTED_TEST_SIZE,
        ),
        flush=True,
    )

    rows = asyncio.run(run_all_modes(test_traces))
    save_ablation_results(rows, OUTPUT_CSV_PATH)

    print("guideline compliance metrics")
    print("mode,n_cases,compliance_rate,hallucination_rate,redundancy_rate,average_steps")
    for row in rows:
        print(
            f"{row['mode']},{int(row['n_cases'])},"
            f"{float(row['compliance_rate']):.4f},"
            f"{float(row['hallucination_rate']):.4f},"
            f"{float(row['redundancy_rate']):.4f},"
            f"{float(row['average_steps']):.4f}"
        )

    print(f"[ablation] Saved metrics to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
