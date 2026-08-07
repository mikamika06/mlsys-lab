import numpy as np


def compute_bimodal_summary(latencies_a, latencies_b):
    a = np.array(latencies_a, dtype=np.float64)
    b = np.array(latencies_b, dtype=np.float64)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    median_a = float(np.median(a))
    median_b = float(np.median(b))

    mean_speedup = mean_a / mean_b if mean_b > 0 else 0.0
    median_speedup = median_a / median_b if median_b > 0 else 0.0

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "median_a": median_a,
        "median_b": median_b,
        "mean_speedup": mean_speedup,
        "median_speedup": median_speedup,
    }


def bootstrap_paired_speedup_ci(latencies_a, latencies_b, num_resamples=1000, ci_level=0.95, seed=42):
    a = np.array(latencies_a, dtype=np.float64)
    b = np.array(latencies_b, dtype=np.float64)

    n = len(a)
    rng = np.random.default_rng(seed)

    idxs = rng.integers(0, n, size=(num_resamples, n))
    resampled_a = a[idxs]
    resampled_b = b[idxs]

    speedups = np.mean(resampled_a, axis=1) / np.mean(resampled_b, axis=1)

    alpha = (1.0 - ci_level) / 2.0
    low_p = alpha * 100.0
    high_p = (1.0 - alpha) * 100.0

    ci_low = float(np.percentile(speedups, low_p))
    ci_high = float(np.percentile(speedups, high_p))
    point_estimate = float(np.mean(a) / np.mean(b))

    return {
        "point_estimate": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
