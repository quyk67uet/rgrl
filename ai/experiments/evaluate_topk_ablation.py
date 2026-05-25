"""Sweep Guidance top-k and report retrieval coverage.

This script evaluates Dual-Encoder guidance quality for top-k in {1,3,5,8}.
For each k it computes:
- fraction of traces where the ground-truth next agent is present in top-k guidance
- fraction where AFAN-GNN's predicted action maps to an agent present in top-k

Outputs:
- ai/results/ablation_topk.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from ai.experiments.evaluate_tradeoff import load_test_traces
from ai.dual_system.neuro_symbolic_router import _get_system1_engine

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "ai" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Agent ordering used across envs
AGENT_NAMES = [
    "TriageAgent",
    "EHRAgent",
    "DispensationAgent",
    "ReconciliationAgent",
    "SummaryAgent",
]


def run_topk_ablation(k_values=(1, 3, 5, 8)) -> List[Dict[str, Any]]:
    traces = load_test_traces()
    results: List[Dict[str, Any]] = []

    # Try to import DualGuidanceMechanism; if unavailable skip guidance metrics
    try:
        from ai.gro.scripts.guidance_dual import DualGuidanceMechanism
        guidance_available = True
    except Exception:
        DualGuidanceMechanism = None  # type: ignore
        guidance_available = False

    # AFAN engine (may raise if torch missing)
    engine = _get_system1_engine()
    try:
        engine._ensure_loaded()
        afan_available = True
    except Exception:
        afan_available = False

    for k in k_values:
        gt_hits = 0
        afan_hits = 0
        n = len(traces)

        guidance = None
        if guidance_available:
            # Use default embedding space path if available; callers may edit code to point elsewhere
            emb_path = str(REPO_ROOT / "ai" / "models" / "stage1_wrl_encoders")
            try:
                guidance = DualGuidanceMechanism(state_encoder_path="all-mpnet-base-v2", action_encoder_path="all-mpnet-base-v2", embedding_space_path=emb_path)
            except Exception:
                guidance = None

        for trace in traces:
            # Derive current state as initial orchestrator input_state
            spans = trace.get("spans", [])
            initial_state = None
            gt_first_agent = None
            for span in spans:
                attrs = span.get("attributes", {})
                if attrs.get("vhas.span.type") == "orchestrator_decision":
                    if initial_state is None:
                        initial_state = attrs.get("vhas.orchestrator.input_state")
                        gt_first_agent = attrs.get("vhas.orchestrator.action_selected")
                        break

            if not initial_state or not gt_first_agent:
                continue

            proposed_agents: List[str] = []
            if guidance is not None:
                try:
                    proposed_agents = guidance.propose_actions(initial_state, top_k=k)
                except Exception:
                    proposed_agents = []

            if gt_first_agent in proposed_agents:
                gt_hits += 1

            if afan_available:
                try:
                    action_payload, conf = engine.infer(patient_context=initial_state or "", graph_history=trace.get("graph_history", []))
                    predicted_idx = action_payload.get("predicted_action")
                    if isinstance(predicted_idx, int) and 0 <= predicted_idx < len(AGENT_NAMES):
                        predicted_agent = AGENT_NAMES[predicted_idx]
                        if predicted_agent in proposed_agents:
                            afan_hits += 1
                except Exception:
                    pass

        results.append({
            "top_k": k,
            "n_traces": n,
            "ground_truth_in_topk_rate": gt_hits / float(max(1, n)),
            "afan_in_topk_rate": afan_hits / float(max(1, n)),
        })

    return results


def save_results(rows: List[Dict[str, Any]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> None:
    rows = run_topk_ablation()
    save_results(rows, RESULTS_DIR / "ablation_topk.csv")


if __name__ == "__main__":
    main()
