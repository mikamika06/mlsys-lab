import numpy as np
from mlsys.scorers import rel_err

def _ref(op, dims):
    """Oracle: recompute flops, bytes, ai from the canonical formulas."""
    BYTES_PER_ELEM = 8  # FP64

    if op == "saxpy":
        n = dims["n"]
        flops = 2 * n
        byte_count = 3 * n * BYTES_PER_ELEM          # read X, read Y, write Y
    elif op == "gemm":
        m, k, n = dims["m"], dims["k"], dims["n"]
        flops = 2 * m * n * k
        byte_count = (m * k + k * n + m * n) * BYTES_PER_ELEM
    else:
        raise ValueError(f"unknown op {op}")

    ai = flops / byte_count
    return float(flops), float(byte_count), float(ai)

def grade(sol, fx) -> dict:
    cases = [
        ("saxpy", {"n": 1024}),
        ("saxpy", {"n": 1_000_000}),
        ("gemm", {"m": 128, "k": 256, "n": 512}),
        ("gemm", {"m": 3, "k": 4, "n": 5}),
        ("gemm", {"m": 1, "k": 1, "n": 1}),
    ]

    got_flops, got_bytes, got_ai = [], [], []
    ref_flops, ref_bytes, ref_ai = [], [], []

    for op, dims in cases:
        try:
            r = sol.compute_roofline_metrics(op, **dims)
            got_flops.append(float(r["flops"]))
            got_bytes.append(float(r["bytes"]))
            got_ai.append(float(r["ai"]))
        except Exception:
            # Student function failed — return max error for every gate.
            return {
                "flops_rel_err": 1.0,
                "bytes_rel_err": 1.0,
                "ai_rel_err": 1.0,
            }

        rf, rb, ra = _ref(op, dims)
        ref_flops.append(rf)
        ref_bytes.append(rb)
        ref_ai.append(ra)

    ref_f = np.array(ref_flops, dtype=np.float64)
    ref_b = np.array(ref_bytes, dtype=np.float64)
    ref_a = np.array(ref_ai, dtype=np.float64)
    got_f = np.array(got_flops, dtype=np.float64)
    got_b = np.array(got_bytes, dtype=np.float64)
    got_a = np.array(got_ai, dtype=np.float64)

    return {
        "flops_rel_err": rel_err(ref_f, got_f),
        "bytes_rel_err": rel_err(ref_b, got_b),
        "ai_rel_err": rel_err(ref_a, got_a),
    }
