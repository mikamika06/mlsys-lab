from typing import Dict, Any


def compute_recovery_percentage(
    baseline_score: float,
    quantized_score: float,
    random_score: float
) -> float:
    """Compute recovery percentage relative to baseline and random performance."""
    denom = baseline_score - random_score
    if abs(denom) < 1e-12:
        return 100.0 if quantized_score >= baseline_score else 0.0
    rec = ((quantized_score - random_score) / denom) * 100.0
    return float(rec)


def parse_lm_eval_recovery(
    eval_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    random_baseline: float = 0.0
) -> Dict[str, float]:
    """Parse lm-eval JSON dicts and compute per-task recovery percentages."""
    eval_results_data = eval_results.get("results", {})
    base_results_data = baseline_results.get("results", {})

    out = {}
    for task_name, task_data in eval_results_data.items():
        if task_name not in base_results_data:
            continue

        q_metric = None
        b_metric = None

        for k in ("acc_norm,none", "acc,none", "acc_norm", "acc", "exact_match,none", "exact_match"):
            if k in task_data and k in base_results_data[task_name]:
                q_metric = task_data[k]
                b_metric = base_results_data[task_name][k]
                break

        if q_metric is None:
            continue

        rec = compute_recovery_percentage(b_metric, q_metric, random_baseline)
        out[task_name] = rec

    return out
