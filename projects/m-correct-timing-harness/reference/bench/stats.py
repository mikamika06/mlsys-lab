"""Statistical calculations for benchmark sample size and confidence intervals."""
import numpy as np


Z_VALUES = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576
}


def required_sample_size(samples, target_rel_ci=0.05, confidence=0.95):
    """Calculates the minimum required sample size to reach the target relative confidence interval width."""
    arr = np.array(samples, dtype=np.float64)
    if len(arr) < 2:
        return 30

    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if mean == 0:
        return 0

    cv = std / mean
    z = Z_VALUES.get(confidence, 1.960)

    n_required = (z * cv / target_rel_ci) ** 2
    return int(np.ceil(n_required))


def compute_ci_bounds(samples, confidence=0.95):
    """Computes mean, standard error, and relative CI error bound for a sample array."""
    arr = np.array(samples, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std_err": 0.0, "rel_ci": 0.0}

    mean = float(np.mean(arr))
    if n == 1:
        return {"mean": mean, "std_err": 0.0, "rel_ci": 0.0}

    std = float(np.std(arr, ddof=1))
    std_err = std / np.sqrt(n)

    z = Z_VALUES.get(confidence, 1.960)
    margin_of_error = z * std_err
    rel_ci = margin_of_error / mean if mean != 0 else 0.0

    return {
        "mean": mean,
        "std_err": float(std_err),
        "rel_ci": float(rel_ci)
    }
