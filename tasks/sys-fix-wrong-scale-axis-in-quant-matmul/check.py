import numpy as np

def _ref(x, codes, scales):
    """Correct quantized matmul: per-output-channel scales along axis 1."""
    W = codes.astype(np.float64) * scales          # (K,N)*(N,) -> column scaling
    return x @ W

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(42)
    cases = [
        (4, 8, 16),
        (1, 32, 64),
        (8, 16, 4),
        (16, 128, 128),
        (2, 8, 32),
    ]
    max_err = 0.0
    for M, K, N in cases:
        x = rng.randn(M, K)
        codes = rng.randint(-128, 128, size=(K, N)).astype(np.int8)
        scales = rng.uniform(0.001, 0.1, size=(N,))
        ref = _ref(x, codes, scales)
        try:
            got = np.asarray(sol.quant_matmul(x, codes, scales),
                             dtype=np.float64)
        except Exception:
            return {"rel_err": 1.0}
        err = float(np.linalg.norm(got - ref)
                    / (np.linalg.norm(ref) + 1e-12))
        max_err = max(max_err, err)
    return {"rel_err": max_err}
