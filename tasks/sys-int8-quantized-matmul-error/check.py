import numpy as np


def _oracle(A, B):
    def quantize(x):
        scale = np.max(np.abs(x)) / 127.0
        if scale == 0:
            scale = 1.0
        q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
        return q, scale

    qa, sa = quantize(A)
    qb, sb = quantize(B)
    acc = qa.astype(np.int32) @ qb.astype(np.int32)
    return acc.astype(np.float64) * (sa * sb)


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(1234)
    cases = [
        (rng.normal(0, 1, (4, 8)), rng.normal(0, 1, (8, 3))),
        (rng.normal(0, 5, (16, 32)), rng.normal(0, 2, (32, 7))),
        (rng.normal(0, 0.1, (3, 64)), rng.normal(0, 10, (64, 5))),
    ]

    worst = 0.0
    try:
        for A, B in cases:
            got = sol.quantized_matmul(A, B)
            ref = _oracle(A, B)
            worst = max(worst, _rel_err(got, ref))
    except Exception:
        return {"rel_err": float("inf")}

    return {"rel_err": worst}
