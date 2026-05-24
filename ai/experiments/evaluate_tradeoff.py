"""Real tau-sweep evaluation for the neuro-symbolic dual-system.

This script runs an end-to-end tradeoff experiment over real workflow traces by
calling the production router. It reports:
- System 2 invocation rate
- Compliance rate against ground-truth pathway labels

"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from ai.dual_system.neuro_symbolic_router import orchestrate_workflow_step

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_TRACE_ROOT = REPO_ROOT / "ai" / "data" / "workflow_traces"
TEST_TRACE_PATH = WORKFLOW_TRACE_ROOT / "test_traces.json"
OUTPUT_CSV_PATH = REPO_ROOT / "ai" / "results" / "real_tradeoff_metrics.csv"
EXPECTED_TEST_SIZE = 200
TAU_VALUES = [0.50, 0.70, 0.85, 0.95, 0.99]

_ALLOWED_SKELETONS_TO_PATHWAY: dict[tuple[str, ...], int] = {
    ("TriageAgent", "EHRAgent", "SummaryAgent"): 1,
    ("TriageAgent", "DispensationAgent", "SummaryAgent"): 2,
    (
        "TriageAgent",
        "EHRAgent",
        "DispensationAgent",
        "EHRAgent",
        "ReconciliationAgent",
        "SummaryAgent",
    ): 3,
}


def _read_json_file(path: Path) -> Any:
    """Read a JSON file and return the parsed object."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_agent_sequence_from_spans(spans: list[dict[str, Any]]) -> list[str]:
    """Extract the executed agent sequence from trace spans."""
    sequence: list[str] = []
    for span in spans:
        attributes = span.get("attributes", {})
        if attributes.get("vhas.span.type") == "agent_execution":
            agent_name = attributes.get("vhas.agent.name")
            if isinstance(agent_name, str) and agent_name:
                sequence.append(agent_name)
    return sequence


def _derive_selected_pathway(trace_like: dict[str, Any]) -> int | None:
    """Resolve ground-truth selected pathway from explicit field or agent sequence."""
    explicit_pathway = trace_like.get("selected_pathway")
    if isinstance(explicit_pathway, int):
        return explicit_pathway

    if isinstance(explicit_pathway, str) and explicit_pathway.strip().isdigit():
        return int(explicit_pathway.strip())

    sequence = trace_like.get("ground_truth_agent_sequence")
    if not isinstance(sequence, list) or not sequence:
        spans = trace_like.get("spans", [])
        if isinstance(spans, list):
            sequence = _extract_agent_sequence_from_spans(spans)
        else:
            sequence = []

    sequence_tuple = tuple(str(item) for item in sequence)
    return _ALLOWED_SKELETONS_TO_PATHWAY.get(sequence_tuple)


def _normalize_trace(trace_like: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize raw trace objects into a stable schema used by experiments."""
    if not isinstance(trace_like, dict):
        return None

    patient_context = trace_like.get("patient_context")
    if not isinstance(patient_context, str) or not patient_context.strip():
        scenario = trace_like.get("input_scenario", {})
        if isinstance(scenario, dict):
            patient_context = scenario.get("nl_command")

    if not isinstance(patient_context, str) or not patient_context.strip():
        return None

    spans = trace_like.get("spans", [])
    if not isinstance(spans, list):
        spans = []

    ground_truth_sequence = trace_like.get("ground_truth_agent_sequence")
    if not isinstance(ground_truth_sequence, list) or not ground_truth_sequence:
        ground_truth_sequence = _extract_agent_sequence_from_spans(spans)

    selected_pathway = _derive_selected_pathway(
        {
            "selected_pathway": trace_like.get("selected_pathway"),
            "ground_truth_agent_sequence": ground_truth_sequence,
            "spans": spans,
        }
    )

    if selected_pathway is None:
        return None

    trace_id = trace_like.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id:
        trace_id = f"trace_{abs(hash(patient_context))}"

    return {
        "trace_id": trace_id,
        "patient_context": patient_context.strip(),
        "graph_history": trace_like.get("graph_history", spans),
        "selected_pathway": selected_pathway,
        "ground_truth_agent_sequence": [str(item) for item in ground_truth_sequence],
    }


def load_test_traces() -> list[dict[str, Any]]:
    """Load normalized traces from split-based `test_traces.json`.

    Raises:
        FileNotFoundError: If split files are not prepared yet.
        ValueError: If the test file contains no valid traces.
    """
    if not TEST_TRACE_PATH.exists():
        raise FileNotFoundError(
            "Missing test split file at "
            f"{TEST_TRACE_PATH}. Run `ai/data/workflow_traces/prepare_splits.py` first."
        )

    payload = _read_json_file(TEST_TRACE_PATH)
    if isinstance(payload, dict):
        candidate_traces = payload.get("traces", [])
    elif isinstance(payload, list):
        candidate_traces = payload
    else:
        candidate_traces = []

    normalized: list[dict[str, Any]] = []
    for item in candidate_traces:
        normalized_trace = _normalize_trace(item)
        if normalized_trace is not None:
            normalized.append(normalized_trace)

    if not normalized:
        raise ValueError(f"No valid traces were loaded from {TEST_TRACE_PATH}.")

    return normalized


def _extract_selected_pathway_from_action(action: Any) -> int | None:
    """Extract selected pathway from model action payload."""
    if not isinstance(action, dict):
        return None

    pathway = action.get("selected_pathway")
    if isinstance(pathway, int):
        return pathway
    if isinstance(pathway, str) and pathway.strip().isdigit():
        return int(pathway.strip())
    return None


async def evaluate_tau(
    tau_threshold: float,
    traces: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Evaluate one tau threshold over real traces using the production router."""
    sys2_count = 0
    compliance_count = 0
    total = len(traces)
    semaphore = asyncio.Semaphore(5)

    async def _run_with_semaphore(index: int, trace: dict[str, Any]) -> dict[str, Any]:
        print(
            f"[tradeoff] tau={tau_threshold:.2f} | Evaluating Trace {index}/{total}...",
            flush=True,
        )
        async with semaphore:
            return await orchestrate_workflow_step(
                patient_context=str(trace["patient_context"]),
                graph_history=trace.get("graph_history", []),
                tau_threshold=tau_threshold,
            )

    tasks = [
        asyncio.create_task(_run_with_semaphore(index, trace))
        for index, trace in enumerate(traces, start=1)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for trace, result in zip(traces, results):
        if isinstance(result, Exception):  # pragma: no cover - runtime/network dependent.
            print(
                f"[tradeoff] trace={trace.get('trace_id')} failed with error: {result}",
                flush=True,
            )
            continue

        if result.get("selected_system") == "system2":
            sys2_count += 1

        predicted_pathway = _extract_selected_pathway_from_action(result.get("action"))
        ground_truth_pathway = trace.get("selected_pathway")

        if isinstance(ground_truth_pathway, int) and predicted_pathway == ground_truth_pathway:
            compliance_count += 1

    return {
        "tau_threshold": tau_threshold,
        "n_traces": total,
        "sys2_invocation_rate": sys2_count / float(total),
        "guideline_compliance_rate": compliance_count / float(total),
    }


def save_tradeoff_metrics(rows: list[dict[str, float | int]], output_path: Path) -> Path:
    """Save tradeoff experiment metrics as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tau_threshold",
                "n_traces",
                "sys2_invocation_rate",
                "guideline_compliance_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


async def run_tradeoff_evaluation() -> list[dict[str, float | int]]:
    """Run the full tau sweep and return all metric rows."""
    traces = load_test_traces()
    print(
        "[tradeoff] Loaded {count} traces from {path} (thesis target test size: {target}).".format(
            count=len(traces),
            path=TEST_TRACE_PATH,
            target=EXPECTED_TEST_SIZE,
        ),
        flush=True,
    )

    rows: list[dict[str, float | int]] = []
    for tau in TAU_VALUES:
        row = await evaluate_tau(tau_threshold=tau, traces=traces)
        rows.append(row)
        print(
            "[tradeoff] Completed tau={tau:.2f} | sys2={sys2:.4f} | compliance={comp:.4f}".format(
                tau=tau,
                sys2=float(row["sys2_invocation_rate"]),
                comp=float(row["guideline_compliance_rate"]),
            ),
            flush=True,
        )

    save_tradeoff_metrics(rows=rows, output_path=OUTPUT_CSV_PATH)
    print(f"[tradeoff] Saved metrics to: {OUTPUT_CSV_PATH}", flush=True)
    return rows


def main() -> None:
    """CLI entrypoint for real tradeoff evaluation."""
    rows = asyncio.run(run_tradeoff_evaluation())

    print("tau_threshold,n_traces,sys2_invocation_rate,guideline_compliance_rate")
    for row in rows:
        print(
            f"{float(row['tau_threshold']):.2f},"
            f"{int(row['n_traces'])},"
            f"{float(row['sys2_invocation_rate']):.4f},"
            f"{float(row['guideline_compliance_rate']):.4f}"
        )


if __name__ == "__main__":
    main()
