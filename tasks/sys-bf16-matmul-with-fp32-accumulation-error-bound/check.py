import numpy as np


def _bf16_round(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    bits = (bits + np.uint32(0x8000)) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def _bf16_accum_matmul(a, b):
    a = _bf16_round(a)
    b = _bf16_round(b)
    m, k = a.shape
    _, n = b.shape
    out = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            s = np.float32(0)
            for r in range(k):
                s = _bf16_round(np.float32(s + np.float32(a[i, r] * b[r, j])))
            out[i, j] = s
    return out


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(17)
    cases = [
        (
            rng.normal(size=(8, 16)).astype(np.float32),
            rng.normal(size=(16, 7)).astype(np.float32),
        ),
        (
            rng.normal(scale=3.0, size=(5, 11)).astype(np.float32),
            rng.normal(scale=0.5, size=(11, 9)).astype(np.float32),
        ),
        (
            np.linspace(-2, 2, 24, dtype=np.float32).reshape(4, 6),
            np.linspace(1, -1, 18, dtype=np.float32).reshape(6, 3),
        ),
    ]

    worst = 0.0
    for a, b in cases:
        try:
            got = np.asarray(sol.bf16_matmul_fp32_accum(a, b), dtype=np.float32)
        except Exception:
            return {"rel_err": 1.0}

        oracle = a.astype(np.float64) @ b.astype(np.float64)
        err = np.linalg.norm(got.astype(np.float64) - oracle) / (
            np.linalg.norm(oracle) + 1e-12
        )
        worst = max(worst, float(err))

        low_precision = _bf16_accum_matmul(a, b)
        low_err = np.linalg.norm(low_precision.astype(np.float64) - oracle) / (
            np.linalg.norm(oracle) + 1e-12
        )
        if err >= low_err:
            return {"rel_err": 1.0}

        expected = np.matmul(_bf16_round(a), _bf16_round(b))
        if got.shape != expected.shape:
            return {"rel_err": 1.0}

    return {"rel_err": worst}
