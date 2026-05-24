"""Neuro-symbolic router bridging AFAN-GNN System 1 and deliberative System 2.

This module provides production-oriented routing logic:
1. Run System 1 AFAN-GNN inference on graph history.
2. Compute confidence as max softmax probability over action logits.
3. Delegate to System 2 only when confidence falls below threshold.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .deliberative_llm import run_deliberative_pipeline

if TYPE_CHECKING:
    import torch
    from torch_geometric.data import HeteroData

MODEL_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "stage2_gro_policies"
    / "vhas_gnn_afan_best.pt"
)

# Structural pathway mapping to preserve downstream metric compatibility.
# AFAN predicts an action index; this maps the predicted action to one of three
# structural pathways used by evaluation scripts.
_ACTION_TO_PATHWAY = {
    0: 1,  # Triage-oriented structural start
    1: 1,  # EHR-oriented monitoring branch
    2: 2,  # Dispensation-oriented intervention branch
    3: 3,  # Reconciliation indicates complex care loop
    4: 1,  # Summary default branch for low-acuity closure
}

_model_load_lock = asyncio.Lock()


class AFANGNNInferenceEngine:
    """Runtime inference engine for AFAN-GNN System 1 policy."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self._policy: Any | None = None
        self._device: Any | None = None

    @staticmethod
    def _require_torch() -> Any:
        """Import torch at runtime and return module object."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "System 1 inference requires PyTorch. Install `torch` to enable AFAN-GNN routing."
            ) from exc
        return torch

    @staticmethod
    def _require_heterodata() -> Any:
        """Import PyG HeteroData at runtime and return class object."""
        try:
            from torch_geometric.data import HeteroData
        except ImportError as exc:
            raise RuntimeError(
                "System 1 inference requires torch-geometric with HeteroData support."
            ) from exc
        return HeteroData

    @staticmethod
    def _infer_num_actions_from_state_dict(state_dict: dict[str, Any]) -> int:
        """Infer AFAN action-space size from actor head weight tensor shape."""
        for key, value in state_dict.items():
            if not hasattr(value, "shape"):
                continue
            if "actor" in key and key.endswith("weight") and len(value.shape) == 2:
                out_features = int(value.shape[0])
                if out_features > 1:
                    return out_features

        raise RuntimeError(
            "Unable to infer action-space size from checkpoint state_dict. "
            "Expected actor head weight tensor to be present."
        )

    def _build_policy_instance(self, num_actions: int) -> Any:
        """Instantiate AFAN policy architecture for checkpoint loading."""
        torch = self._require_torch()

        try:
            from tensordict import TensorDict
        except ImportError as exc:
            raise RuntimeError(
                "System 1 inference requires tensordict for AFAN policy construction."
            ) from exc

        try:
            from ai.gro.gnn.actor_critic_gnn import ActorCriticGNN
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import AFAN ActorCriticGNN from ai.gro.gnn.actor_critic_gnn."
            ) from exc

        dummy_obs = TensorDict({}, batch_size=[1])
        policy = ActorCriticGNN(obs=dummy_obs, obs_groups={}, num_actions=num_actions)
        policy.to(self._device)
        policy.eval()
        return policy

    def _ensure_loaded(self) -> None:
        """Lazily load AFAN-GNN policy checkpoint into memory."""
        if self._policy is not None:
            return

        if not self.checkpoint_path.exists():
            raise RuntimeError(
                "AFAN-GNN checkpoint not found at "
                f"{self.checkpoint_path}. Ensure model artifacts are available."
            )

        torch = self._require_torch()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_payload = torch.load(self.checkpoint_path, map_location=self._device)

        if isinstance(checkpoint_payload, torch.nn.Module):
            self._policy = checkpoint_payload.to(self._device).eval()
            return

        if not isinstance(checkpoint_payload, dict):
            raise RuntimeError(
                "Unsupported checkpoint payload for AFAN-GNN. "
                "Expected nn.Module or dict with state_dict keys."
            )

        state_dict_candidate = checkpoint_payload.get("model_state_dict")
        if not isinstance(state_dict_candidate, dict):
            state_dict_candidate = checkpoint_payload.get("state_dict")
        if not isinstance(state_dict_candidate, dict):
            state_dict_candidate = checkpoint_payload

        num_actions = self._infer_num_actions_from_state_dict(state_dict_candidate)
        policy = self._build_policy_instance(num_actions=num_actions)

        load_result = policy.load_state_dict(state_dict_candidate, strict=False)
        missing_keys = list(getattr(load_result, "missing_keys", []))
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []))

        # Require actor head weights to be loaded for valid action inference.
        actor_weights_loaded = any(
            key.startswith("actor") and "weight" in key
            for key in state_dict_candidate.keys()
        )
        if not actor_weights_loaded:
            raise RuntimeError(
                "Checkpoint did not provide AFAN actor head weights; cannot run System 1 inference."
            )

        # Allow partial non-critical mismatch, but fail on a completely incompatible graph backbone.
        if len(missing_keys) > 40 and len(unexpected_keys) > 40:
            raise RuntimeError(
                "AFAN checkpoint appears incompatible with deployed policy architecture. "
                f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
            )

        self._policy = policy

    async def _ensure_loaded_async(self) -> None:
        """Lazily load AFAN-GNN policy checkpoint under an async lock."""
        if self._policy is not None:
            return

        async with _model_load_lock:
            if self._policy is None:
                self._ensure_loaded()

    def _to_heterodata(self, graph_history: Any) -> "HeteroData":
        """Convert graph_history payload into a PyG HeteroData object.

        Supports:
        - Direct HeteroData input
        - Dict payload with node features and edge indices
        """
        HeteroData = self._require_heterodata()
        torch = self._require_torch()

        if isinstance(graph_history, HeteroData):
            data = graph_history
            return data.to(self._device)

        if not isinstance(graph_history, dict):
            raise RuntimeError(
                "graph_history must be a torch_geometric HeteroData or dict convertible to HeteroData."
            )

        data = HeteroData()

        # Node feature conversion
        for node_type in ("state", "agent", "tool"):
            x_key = f"{node_type}_x"
            node_dict = graph_history.get(node_type)

            raw_x = None
            if isinstance(node_dict, dict) and "x" in node_dict:
                raw_x = node_dict["x"]
            elif x_key in graph_history:
                raw_x = graph_history[x_key]

            if raw_x is None:
                continue

            x_tensor = torch.as_tensor(raw_x, dtype=torch.float32, device=self._device)
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0)
            data[node_type].x = x_tensor

        # Edge index conversion
        edge_specs = {
            "state__triggers__agent": ("state", "triggers", "agent"),
            "agent__produces__state": ("agent", "produces", "state"),
            "agent__calls__tool": ("agent", "calls", "tool"),
        }

        for key, edge_type in edge_specs.items():
            edge_payload = graph_history.get(key)
            if edge_payload is None and isinstance(graph_history.get("edges"), dict):
                edge_payload = graph_history["edges"].get(key)

            if edge_payload is None:
                continue

            edge_tensor = torch.as_tensor(edge_payload, dtype=torch.long, device=self._device)
            if edge_tensor.dim() != 2 or edge_tensor.shape[0] != 2:
                raise RuntimeError(
                    f"Invalid edge_index for {key}. Expected shape [2, E], got {tuple(edge_tensor.shape)}."
                )
            data[edge_type].edge_index = edge_tensor

        if not hasattr(data, "x_dict") or len(data.x_dict) == 0:
            raise RuntimeError(
                "Converted HeteroData contains no node features. "
                "Provide state/agent/tool node embeddings in graph_history."
            )

        return data.to(self._device)

    def infer(self, patient_context: str, graph_history: Any) -> tuple[dict[str, Any], float]:
        """Run AFAN-GNN forward pass and return action payload with confidence."""
        self._ensure_loaded()
        if self._policy is None:
            raise RuntimeError("AFAN-GNN policy failed to initialize.")

        torch = self._require_torch()
        from torch_geometric.nn import global_mean_pool

        data = self._to_heterodata(graph_history)

        try:
            with torch.no_grad():
                x_dict = self._policy.gnn(data.x_dict, data.edge_index_dict)
                for node_type, embeddings in x_dict.items():
                    x_dict[node_type] = self._policy.node_norms[node_type](embeddings)

                hidden_dim = int(self._policy.gnn_hidden_dim)
                pooled_parts: list[torch.Tensor] = []
                for node_type in ("agent", "state", "tool"):
                    if node_type in x_dict and x_dict[node_type].numel() > 0:
                        batch_index = torch.zeros(
                            x_dict[node_type].shape[0],
                            dtype=torch.long,
                            device=self._device,
                        )
                        pooled = global_mean_pool(x_dict[node_type], batch_index)
                    else:
                        pooled = torch.zeros((1, hidden_dim), device=self._device)
                    pooled_parts.append(pooled)

                graph_embedding = torch.cat(pooled_parts, dim=-1)
                action_logits = self._policy.actor(graph_embedding)
                action_probs = torch.softmax(action_logits, dim=-1)

                confidence_tensor, action_index_tensor = torch.max(action_probs, dim=-1)
                gnn_confidence = float(confidence_tensor.item())
                predicted_action = int(action_index_tensor.item())

        except Exception as exc:
            raise RuntimeError(f"AFAN-GNN inference failed: {exc}") from exc

        selected_pathway = _ACTION_TO_PATHWAY.get(
            predicted_action,
            max(1, min(3, predicted_action + 1)),
        )

        system1_action = {
            "predicted_action": predicted_action,
            "selected_pathway": selected_pathway,
            "action_probabilities": action_probs.squeeze(0).detach().cpu().tolist(),
            "clinical_rationale": (
                "System 1 AFAN-GNN selected the highest-probability structural action "
                "from graph-encoded workflow context."
            ),
            "source": "system1_gnn_afan",
            "patient_context": patient_context,
        }
        return system1_action, gnn_confidence


_SYSTEM1_ENGINE: AFANGNNInferenceEngine | None = None


def _get_system1_engine() -> AFANGNNInferenceEngine:
    """Get singleton AFAN inference engine for process-wide reuse."""
    global _SYSTEM1_ENGINE
    if _SYSTEM1_ENGINE is None:
        _SYSTEM1_ENGINE = AFANGNNInferenceEngine(checkpoint_path=MODEL_CHECKPOINT_PATH)
    return _SYSTEM1_ENGINE


async def orchestrate_workflow_step(
    patient_context: str,
    graph_history: Any,
    tau_threshold: float = 0.85,
) -> dict[str, Any]:
    """Route one workflow step through System 1 AFAN-GNN or System 2 reasoning.

    Args:
        patient_context: Current patient context in natural language.
        graph_history: Historical workflow graph input for System 1.
        tau_threshold: Confidence threshold for structural delegation.

    Returns:
        dict[str, Any]: Routing decision and selected action payload.

    Raises:
        ValueError: If inputs are invalid.
        RuntimeError: If System 1 model loading or inference fails.
    """
    if not patient_context or not patient_context.strip():
        raise ValueError("patient_context must be a non-empty string.")
    if not 0.0 <= tau_threshold <= 1.0:
        raise ValueError("tau_threshold must be within [0.0, 1.0].")

    system1_policy = _get_system1_engine()
    await system1_policy._ensure_loaded_async()
    gnn_action, gnn_confidence = system1_policy.infer(
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