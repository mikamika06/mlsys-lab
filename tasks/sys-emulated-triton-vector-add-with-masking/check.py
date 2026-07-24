import numpy as np

def _oracle_add(a, b):
    """Ground-truth: plain element-wise addition."""
    return a + b

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(42)

    cases = [
        (5,   2,   "small, partial last block"),
        (10,  3,   "N not divisible by B"),
        (7,   4,   "odd N, block > N/2"),
        (1,   16,  "N < block_size"),
        (0,   32,  "empty input"),
        (100, 16,  "medium random"),
        (1000, 64, "large random"),
        (17,  8,   "prime-ish N vs block"),
        (64,  8,   "exact divisibility"),
        (128, 50,  "non-power-of-two block"),
    ]

    max_err = 0.0

    for N, B, desc in cases:
        a = rng.randn(N).astype(np.float64)
        b = rng.randn(N).astype(np.float64)
        expected = _oracle_add(a, b)

        try:
            got = sol.emulated_triton_add(a.copy(), b.copy(), B)
        except Exception:
            return {"max_abs_err": float("inf")}

        if not isinstance(got, np.ndarray):
            return {"max_abs_err": float("inf")}

        if got.shape != expected.shape:
            return {"max_abs_err": float("inf")}

        if got.dtype != expected.dtype:
            return {"max_abs_err": float("inf")}

        if got.size == 0:
            case_err = 0.0
        else:
            case_err = float(np.max(np.abs(got - expected)))

        if case_err > max_err:
            max_err = case_err

    return {"max_abs_err": max_err}
