import numpy as np


def compute_speedup(acceptance_trace, k, draft_cost, target_cost):
    """Compute realized vs theoretical speedup from an acceptance trace."""
    trace = np.asarray(acceptance_trace, dtype=np.float64)
    total_passes = len(trace)
    if total_passes == 0:
        return {
            "realized_speedup": 0.0,
            "theoretical_speedup": 0.0,
            "mean_acceptance_rate": 0.0,
            "expected_tokens_per_pass": 0.0,
        }

    total_accepted_draft = float(np.sum(trace))
    mean_accepted = total_accepted_draft / total_passes
    alpha = total_accepted_draft / (total_passes * k)

    expected_tokens_per_pass = mean_accepted + 1.0
    c = draft_cost / target_cost

    realized_speedup = expected_tokens_per_pass / (1.0 + k * c)

    if abs(alpha - 1.0) < 1e-9:
        theoretical_tokens = float(k + 1)
    else:
        theoretical_tokens = (1.0 - alpha ** (k + 1)) / (1.0 - alpha)

    theoretical_speedup = theoretical_tokens / (1.0 + k * c)

    return {
        "realized_speedup": float(realized_speedup),
        "theoretical_speedup": float(theoretical_speedup),
        "mean_acceptance_rate": float(alpha),
        "expected_tokens_per_pass": float(expected_tokens_per_pass),
    }
