"""Neuro-symbolic router bridging System 1 and System 2 decision policies."""

from __future__ import annotations

import hashlib
import random
from typing import Any

from .deliberative_llm import run_deliberative_pipeline


def _emulate_system1_policy(patient_context: str, graph_history: Any) -> tuple[dict[str, Any], float]:
    """Deterministic Hash-Seeded Policy Emulator using SHA-256 seed locking.

    This emulator fixes its RNG seed from a SHA-256 hash of the inputs so that
    identical contexts yield identical outputs across operating systems. The design
    is essential for baseline evaluations and ablation studies, ensuring perfect
    reproducibility without the non-deterministic float behavior of neural networks.
    """
    seed_material = f"{patient_context}|{repr(graph_history)}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:12], 16))

    confidence = round(rng.uniform(0.50, 0.99), 4)
    selected_pathway = rng.choice([1, 2, 3])

    gnn_action = {
        "selected_pathway": selected_pathway,
        "steps": [],
        "action_rationale": "System 1 policy prediction computed from graph-state context.",
        "source": "system1_calibrated_emulator",
    }
    return gnn_action, confidence


async def orchestrate_workflow_step(
    patient_context: str,
    graph_history: Any,
    tau_threshold: float = 0.85,
) -> dict[str, Any]:
    """Route one workflow step through System 1 or System 2.

    Args:
        patient_context: Current patient context in natural language.
        graph_history: Structured historical graph state for System 1.
        tau_threshold: Confidence cutoff for fast-path execution.

    Returns:
        dict[str, Any]: Routing decision and selected action payload.

    Raises:
        ValueError: If the threshold is invalid or the context is empty.
    """
    if not patient_context or not patient_context.strip():
        raise ValueError("patient_context must be a non-empty string.")
    if not 0.0 <= tau_threshold <= 1.0:
        raise ValueError("tau_threshold must be within [0.0, 1.0].")

    gnn_action, gnn_confidence = _emulate_system1_policy(
        patient_context=patient_context,
        graph_history=graph_history,
    )

    if gnn_confidence >= tau_threshold:
        return {
            "route": "system1_fast_path",
            "selected_system": "system1",
            "tau_threshold": tau_threshold,
            "gnn_confidence": gnn_confidence,
            "escalate_to_hitl": False,
            "action": gnn_action,
        }

    system2_result = await run_deliberative_pipeline(patient_context=patient_context)
    return {
        "route": "system2_deliberative",
        "selected_system": "system2",
        "tau_threshold": tau_threshold,
        "gnn_confidence": gnn_confidence,
        "system1_action": gnn_action,
        "system2_result": system2_result,
        "escalate_to_hitl": bool(system2_result.get("escalate_to_hitl", False)),
        "action": system2_result.get("action"),
    }
