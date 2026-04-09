"""Experiment runners for dual-system tradeoff and ablation studies."""

from .evaluate_ablation import run_ablation_experiment
from .evaluate_tradeoff import TradeoffEvaluator

__all__ = ["TradeoffEvaluator", "run_ablation_experiment"]
