import numpy as np


def _ref_labels(steps, bytes_per_param, peak_flops, peak_bandwidth):
    steps = np.asarray(steps, dtype=np.float64)
    T = steps[:, 0] + steps[:, 1]
    ai = 2.0 * T / bytes_per_param
    ai_ridge = peak_flops / peak_bandwidth
    compute = ai >= ai_ridge
    return np.where(compute, "compute", "memory")


def _scenarios():
    scenarios = []

    # bytes_per_param=2.0 (fp16), ridge = 200e12/4e12 = 50 -> T_threshold = 50
    hw1 = dict(bytes_per_param=2.0, peak_flops=200e12, peak_bandwidth=4e12)
    steps1 = [
        (50, 0),    # T=50, AI=50 == ridge -> compute (boundary, inclusive)
        (49, 0),    # T=49, AI=49 < ridge -> memory
        (51, 0),    # T=51, AI=51 > ridge -> compute
        (0, 50),    # same T via prefill only -> compute
        (25, 25),   # mixed, T=50 -> compute
        (1, 0),     # tiny decode-only -> memory
        (0, 0),     # degenerate zero-token step -> memory (AI=0 < ridge)
        (0, 1000),  # big prefill chunk -> compute
    ]
    scenarios.append((steps1, hw1))

    # bytes_per_param=1.0 (int8), peak_flops=100e12, peak_bandwidth=5e12 -> ridge=20 -> T_threshold=10
    hw2 = dict(bytes_per_param=1.0, peak_flops=100e12, peak_bandwidth=5e12)
    steps2 = [
        (10, 0),   # T=10, AI=20 == ridge -> compute
        (9, 0),    # T=9, AI=18 < ridge -> memory
        (3, 8),    # T=11 -> compute
        (2, 2),    # T=4 -> memory
    ]
    scenarios.append((steps2, hw2))

    # seeded random batch
    rng = np.random.default_rng(0)
    hw3 = dict(bytes_per_param=2.0, peak_flops=312e12, peak_bandwidth=3.35e12)
    decode = rng.integers(0, 64, size=200)
    prefill = rng.integers(0, 512, size=200)
    steps3 = list(zip(decode.tolist(), prefill.tolist()))
    scenarios.append((steps3, hw3))

    return scenarios


def grade(sol, fx) -> dict:
    total = 0
    correct = 0

    for steps, hw in _scenarios():
        ref = _ref_labels(steps, hw["bytes_per_param"], hw["peak_flops"], hw["peak_bandwidth"])
        try:
            got = sol.classify_steps(
                [tuple(s) for s in steps],
                hw["bytes_per_param"], hw["peak_flops"], hw["peak_bandwidth"],
            )
        except Exception:
            total += len(steps)
            continue

        try:
            if len(got) != len(ref):
                total += len(steps)
                continue
            for g, r in zip(got, ref):
                total += 1
                if str(g) == str(r):
                    correct += 1
        except Exception:
            total += len(steps)
            continue

    exact_match = (correct / total) if total else 0.0
    return {"exact_match": exact_match}
