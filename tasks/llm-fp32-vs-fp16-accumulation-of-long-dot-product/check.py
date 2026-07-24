import numpy as np


def _fp16_accum(a, b):
    acc = np.float16(0.0)
    for x, y in zip(a, b):
        acc = np.float16(acc + np.float16(x * y))
    return float(acc)


def _fp32_accum(a, b):
    acc = np.float32(0.0)
    for x, y in zip(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)):
        acc = np.float32(acc + np.float32(x * y))
    return float(acc)


def _fp64_oracle(a, b):
    return float(np.dot(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))


def _rel_err(x, y):
    return abs(x - y) / (abs(y) + 1e-12)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = [
        (
            rng.normal(0, 0.1, 50000).astype(np.float16),
            rng.normal(0, 0.1, 50000).astype(np.float16),
        ),
        (
            np.ones(20000, dtype=np.float16),
            np.full(20000, 0.001, dtype=np.float16),
        ),
        (
            rng.normal(1, 0.01, 30000).astype(np.float16),
            rng.normal(-1, 0.01, 30000).astype(np.float16),
        ),
    ]

    worst_rel = 0.0
    worst_gap = 0.0

    for a, b in cases:
        try:
            got = float(sol.fp32_dot_sum(a, b))
        except Exception:
            return {"rel_err": 1.0, "fp16_gap": 0.0}

        truth = _fp64_oracle(a, b)
        fp16_error = _rel_err(_fp16_accum(a, b), truth)
        fp32_oracle_error = _rel_err(_fp32_accum(a, b), truth)
        candidate_error = _rel_err(got, truth)

        worst_rel = max(worst_rel, candidate_error)
        worst_gap = max(worst_gap, fp16_error / (fp32_oracle_error + 1e-12))

    return {
        "rel_err": worst_rel,
        "fp16_gap": worst_gap,
    }
