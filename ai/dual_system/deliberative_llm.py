"""Deliberative System 2 pipeline with Generator-Verifier-Reviser control flow.

This module implements the thesis production logic for deliberative reasoning when
System 1 (GNN) confidence is below threshold. The pipeline is intentionally strict:
Generator -> Verifier -> Reviser, repeated up to a bounded number of retries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .prompts import SYS2_GENERATOR_PROMPT, SYS2_REVISER_PROMPT, SYS2_VERIFIER_PROMPT

_GENERATOR_REVISER_MODEL = "gpt-4.1"
_VERIFIER_MODEL = "gpt-5.2"


@dataclass(frozen=True)
class VerificationResult:
    """Outcome returned by the compliance verifier agent."""

    status: str
    feedback: str


def _build_openai_client() -> Any:
    """Create an AsyncOpenAI client with a runtime import guard.

    Returns:
        Any: AsyncOpenAI client instance.

    Raises:
        RuntimeError: If the openai package is unavailable.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai is required for the deliberative pipeline. Install with `pip install openai`."
        ) from exc

    return AsyncOpenAI()


async def _json_chat_completion(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Run a JSON-constrained chat completion call and parse the result."""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    try:
        raw_content = response.choices[0].message.content or "{}"
    except (AttributeError, IndexError) as exc:
        raise ValueError("OpenAI response did not include a valid message payload.") from exc

    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenAI response was not valid JSON: {raw_content}") from exc


def _build_generator_prompt(
    patient_context: str,
    previous_candidate: dict[str, Any] | None,
    previous_feedback: str | None,
) -> str:
    """Construct the user prompt for the Generator agent."""
    blocks = [f"Task context:\n{patient_context}"]

    if previous_candidate is not None:
        blocks.append(
            "Previous candidate solution to improve:\n"
            + json.dumps(previous_candidate, indent=2, sort_keys=True)
        )
    if previous_feedback:
        blocks.append(f"Verifier feedback from prior iteration:\n{previous_feedback}")

    blocks.append("Return a complete candidate solution in JSON.")
    return "\n\n".join(blocks)


async def _generate_candidate(
    client: Any,
    patient_context: str,
    previous_candidate: dict[str, Any] | None,
    previous_feedback: str | None,
) -> dict[str, Any]:
    """Generate a candidate workflow from task context and prior signals."""
    return await _json_chat_completion(
        client=client,
        model=_GENERATOR_REVISER_MODEL,
        system_prompt=SYS2_GENERATOR_PROMPT,
        user_prompt=_build_generator_prompt(
            patient_context=patient_context,
            previous_candidate=previous_candidate,
            previous_feedback=previous_feedback,
        ),
    )


async def _verify_candidate(
    client: Any,
    patient_context: str,
    candidate_solution: dict[str, Any],
) -> VerificationResult:
    """Verify compliance of a candidate workflow against operational guidelines."""
    verifier_input = "\n\n".join(
        [
            f"Task context:\n{patient_context}",
            "Candidate solution JSON:",
            json.dumps(candidate_solution, indent=2, sort_keys=True),
        ]
    )
    raw_verdict = await _json_chat_completion(
        client=client,
        model=_VERIFIER_MODEL,
        system_prompt=SYS2_VERIFIER_PROMPT,
        user_prompt=verifier_input,
    )

    status = str(raw_verdict.get("status", "Flawed")).strip().capitalize()
    if status not in {"Correct", "Flawed"}:
        status = "Flawed"

    feedback = str(raw_verdict.get("feedback", "Verifier did not provide feedback."))
    return VerificationResult(status=status, feedback=feedback)


async def _revise_candidate(
    client: Any,
    patient_context: str,
    candidate_solution: dict[str, Any],
    verifier_feedback: str,
) -> dict[str, Any]:
    """Revise a flawed candidate using verifier feedback."""
    reviser_input = "\n\n".join(
        [
            f"Task context:\n{patient_context}",
            "Flawed candidate solution JSON:",
            json.dumps(candidate_solution, indent=2, sort_keys=True),
            f"Verifier feedback:\n{verifier_feedback}",
        ]
    )

    revised_payload = await _json_chat_completion(
        client=client,
        model=_GENERATOR_REVISER_MODEL,
        system_prompt=SYS2_REVISER_PROMPT,
        user_prompt=reviser_input,
    )

    # Preserve existing keys if the reviser only returns a partial correction.
    merged = dict(candidate_solution)
    merged.update(revised_payload)
    return merged


def _build_hitl_response(
    reason: str,
    attempts: int,
    last_candidate: dict[str, Any] | None,
    last_feedback: str | None,
) -> dict[str, Any]:
    """Build a standardized human-in-the-loop escalation payload."""
    return {
        "status": "hitl_required",
        "selected_system": "system2",
        "escalate_to_hitl": True,
        "attempts": attempts,
        "reason": reason,
        "last_candidate": last_candidate,
        "last_feedback": last_feedback,
    }


async def run_deliberative_pipeline(patient_context: str, max_retries: int = 3) -> dict[str, Any]:
    """Run the deliberative System 2 pipeline for low-confidence cases.

    The control policy follows a bounded iterative loop:
    1. Generator proposes a candidate workflow.
    2. Verifier checks strict workflow compliance.
    3. If flawed, Reviser attempts to correct the candidate.

    Args:
        patient_context: Task context string for the current episode/step.
        max_retries: Maximum number of Generator-Verifier iterations.
            If set to 0, runs Generator-only mode (no Verifier/Reviser).

    Returns:
        dict[str, Any]: Either a verified action payload or a HITL escalation payload.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not patient_context or not patient_context.strip():
        raise ValueError("context input must be a non-empty string.")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0.")

    client = _build_openai_client()

    previous_candidate: dict[str, Any] | None = None
    previous_feedback: str | None = None

    if max_retries == 0:
        try:
            candidate = await _generate_candidate(
                client=client,
                patient_context=patient_context,
                previous_candidate=None,
                previous_feedback=None,
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent.
            return _build_hitl_response(
                reason=f"system2_generator_only_error: {exc}",
                attempts=0,
                last_candidate=None,
                last_feedback=None,
            )

        return {
            "status": "generated_only",
            "selected_system": "system2",
            "escalate_to_hitl": False,
            "attempts": 0,
            "action": candidate,
            "verifier_feedback": None,
        }

    for attempt in range(1, max_retries + 1):
        try:
            candidate = await _generate_candidate(
                client=client,
                patient_context=patient_context,
                previous_candidate=previous_candidate,
                previous_feedback=previous_feedback,
            )
            verdict = await _verify_candidate(
                client=client,
                patient_context=patient_context,
                candidate_solution=candidate,
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent.
            return _build_hitl_response(
                reason=f"system2_runtime_error: {exc}",
                attempts=attempt,
                last_candidate=previous_candidate,
                last_feedback=previous_feedback,
            )

        if verdict.status == "Correct":
            return {
                "status": "verified",
                "selected_system": "system2",
                "escalate_to_hitl": False,
                "attempts": attempt,
                "action": candidate,
                "verifier_feedback": verdict.feedback,
            }

        previous_feedback = verdict.feedback
        previous_candidate = candidate

        if attempt >= max_retries:
            break

        try:
            revised_candidate = await _revise_candidate(
                client=client,
                patient_context=patient_context,
                candidate_solution=candidate,
                verifier_feedback=verdict.feedback,
            )
            revised_verdict = await _verify_candidate(
                client=client,
                patient_context=patient_context,
                candidate_solution=revised_candidate,
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent.
            return _build_hitl_response(
                reason=f"system2_revision_error: {exc}",
                attempts=attempt,
                last_candidate=previous_candidate,
                last_feedback=previous_feedback,
            )

        previous_candidate = revised_candidate
        previous_feedback = revised_verdict.feedback

        if revised_verdict.status == "Correct":
            return {
                "status": "verified",
                "selected_system": "system2",
                "escalate_to_hitl": False,
                "attempts": attempt,
                "action": revised_candidate,
                "verifier_feedback": revised_verdict.feedback,
            }

    return _build_hitl_response(
        reason="max_retries_exceeded",
        attempts=max_retries,
        last_candidate=previous_candidate,
        last_feedback=previous_feedback,
    )
