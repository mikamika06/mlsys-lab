import numpy as np

def _codebook():
    np.random.seed(0)
    sample = np.random.randn(1000000)
    q = (np.arange(16) + 0.5) / 16
    return np.quantile(sample, q)

_CODEBOOK = _codebook()

def _ref_quantize_dequant(x: np.ndarray, blocksize: int = 128):
    arr = np.asarray(x, dtype=np.float64).ravel()
    n = len(arr)
    codes = np.empty_like(arr, dtype=np.uint8)
    deq = np.empty_like(arr, dtype=np.float64)
    for i in range(0, n, blocksize):
        block = arr[i:i+blocksize]
        m = np.max(np.abs(block))
        if m == 0:
            m = 1.0
        y = block / m
        idx = np.abs(y[:, None] - _CODEBOOK).argmin(axis=1)
        codes[i:i+blocksize] = idx.astype(np.uint8)
        deq[i:i+blocksize] = _CODEBOOK[idx] * m
    return codes.reshape(x.shape), deq.reshape(x.shape)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        rng.standard_normal((10,)),
        rng.standard_normal((3, 4)),
        rng.standard_normal((5, 5, 5)),
        rng.integers(-1000, 1000, size=(20,)).astype(np.float64),
        np.zeros((7, 8)),
    ]
    ok_exact = 1.0
    max_err = 0.0
    for arr in cases:
        try:
            codes, deq = sol.nf4_quantize_dequant(arr)
        except Exception:
            return {"exact_match": 0.0, "max_abs_err": float("inf")}
        ref_codes, ref_deq = _ref_quantize_dequant(arr)
        if not np.array_equal(codes, ref_codes):
            ok_exact = 0.0
        err = np.max(np.abs(deq - ref_deq))
        if err > max_err:
            max_err = err
    return {"exact_match": ok_exact, "max_abs_err": max_err}
