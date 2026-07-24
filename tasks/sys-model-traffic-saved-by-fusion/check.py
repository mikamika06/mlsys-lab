import numpy as np


def _oracle(n, k, dtype_bytes):
    bytes_unfused = 2 * k * n * dtype_bytes
    bytes_fused = 2 * n * dtype_bytes
    return bytes_unfused, bytes_fused


def grade(sol, fx) -> dict:
    """
    Builds several seeded random (n, k, dtype_bytes) configs, computes the
    exact reference bytes_unfused = 2*k*n*dtype_bytes and
    bytes_fused = 2*n*dtype_bytes directly, and compares them to the
    submission's fusion_traffic(...) output. Reports the worst-case abs
    error in the fused/unfused ratio vs the analytic 1/k ("size_ratio") and
    the worst-case relative error in the raw byte counts ("rel_err").
    """
    rng = np.random.default_rng(0)
    configs = []
    for _ in range(8):
        n = int(rng.integers(1, 5_000_000))
        k = int(rng.integers(1, 20))
        dtype_bytes = int(rng.choice([1, 2, 4, 8]))
        configs.append((n, k, dtype_bytes))
    configs.append((10_000, 1, 4))  # k=1: no fusion opportunity, ratio == 1

    worst_ratio_err = 0.0
    worst_rel_err = 0.0
    for n, k, dtype_bytes in configs:
        exp_unfused, exp_fused = _oracle(n, k, dtype_bytes)
        exp_ratio = exp_fused / exp_unfused

        try:
            got_unfused, got_fused = sol.fusion_traffic(n, k, dtype_bytes)
            got_unfused = float(got_unfused)
            got_fused = float(got_fused)
        except Exception:
            return {"size_ratio": 1e9, "rel_err": 1e9}

        if got_unfused <= 0:
            return {"size_ratio": 1e9, "rel_err": 1e9}

        got_ratio = got_fused / got_unfused
        worst_ratio_err = max(worst_ratio_err, abs(got_ratio - exp_ratio))
        worst_rel_err = max(
            worst_rel_err,
            abs(got_unfused - exp_unfused) / (abs(exp_unfused) + 1e-12),
            abs(got_fused - exp_fused) / (abs(exp_fused) + 1e-12),
        )

    return {"size_ratio": worst_ratio_err, "rel_err": worst_rel_err}
