import numpy as np


def _oracle(ops, peak_flops, bandwidth):
    flops = np.asarray([op["flops"] for op in ops], dtype=np.float64)
    bytes_accessed = np.asarray([op["bytes"] for op in ops], dtype=np.float64)
    ai = flops / bytes_accessed
    balance = np.float64(peak_flops) / np.float64(bandwidth)
    labels = np.where(ai >= balance, "compute", "memory")
    return [
        {"name": op["name"], "ai": float(a), "bound": str(label)}
        for op, a, label in zip(ops, ai, labels)
    ]


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                {"name": "gemm", "flops": 1_000_000, "bytes": 10_000},
                {"name": "copy", "flops": 1_000, "bytes": 100_000},
                {"name": "conv", "flops": 8_000_000, "bytes": 1_000_000},
            ],
            1e12,
            1e11,
        ),
        (
            [
                {"name": "small", "flops": 512, "bytes": 64},
                {"name": "large_mem", "flops": 100, "bytes": 10_000},
            ],
            2e9,
            1e9,
        ),
        (
            [
                {"name": "boundary", "flops": 5000, "bytes": 100},
            ],
            5e10,
            1e9,
        ),
    ]

    ok = 1.0
    for ops, peak, bandwidth in cases:
        try:
            got = sol.classify_roofline_ops(ops, peak, bandwidth)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(ops, peak, bandwidth):
            ok = 0.0
            break
    return {"exact_match": ok}
