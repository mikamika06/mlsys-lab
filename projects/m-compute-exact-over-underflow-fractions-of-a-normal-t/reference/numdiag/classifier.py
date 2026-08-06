"""Training log anomaly classification module."""

from typing import Dict, List, Union


def classify_training_log_symptoms(log_entries: List[Dict[str, Union[float, str]]]) -> List[str]:
    """Classify 4 training log symptoms into root numerical failure causes."""
    results = []
    for entry in log_entries:
        grad_norm = float(entry.get("grad_norm", 0.0))
        loss = float(entry.get("loss", 0.0))
        loss_delta = float(entry.get("loss_delta", 0.0))
        unique_act_ratio = float(entry.get("unique_activation_ratio", 1.0))
        is_nan_or_inf = entry.get("is_nan_or_inf", False)

        if is_nan_or_inf or loss != loss or grad_norm > 1e4:
            results.append("FP16_OVERFLOW")
        elif grad_norm == 0.0 and loss > 0.0:
            results.append("FP16_UNDERFLOW")
        elif unique_act_ratio < 0.05:
            results.append("REPRESENTATION_COLLAPSE")
        elif abs(loss_delta) < 1e-6 and grad_norm < 1e-5:
            results.append("GRADIENT_VANISHING")
        else:
            results.append("UNKNOWN")
    return results
