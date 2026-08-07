from typing import Dict, Any


def compute_recovery_percentage(
    baseline_score: float,
    quantized_score: float,
    random_score: float
) -> float:
    """Compute recovery percentage relative to baseline and random performance."""
    raise NotImplementedError


def parse_lm_eval_recovery(
    eval_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    random_baseline: float = 0.0
) -> Dict[str, float]:
    """Parse lm-eval JSON dicts and compute per-task recovery percentages."""
    raise NotImplementedError
