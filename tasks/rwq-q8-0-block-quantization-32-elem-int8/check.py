import numpy as np
from mlsys.scorers import rel_err

def _oracle_quantize(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation of Q8_0 block quantization."""
    arr = np.asarray(arr, dtype=np.float64)
    n = arr.size
    codes = np.empty(n, dtype=np.int8)
    deq = np.empty(n, dtype=np.float64)
    block_size = 32
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = arr[start:end]
        amax = np.max(np.abs(block))
        if amax == 0:
            d = 1.0
        else:
            d = amax / 127.0
        q = np.round(block / d).astype(np.int8)
        codes[start:end] = q
        deq[start:end] = q.astype(np.float64) * d
    return codes, deq

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    worst_err = 0.0
    for _ in range(5):
        size = rng.integers(10, 200)
        arr = rng.standard_normal(size).astype(np.float64)
        try:
            codes, deq = sol.q8_0_quantize(arr)
        except Exception:
            return {"rel_err": 1e9}
        ref_codes, ref_deq = _oracle_quantize(arr)

        # Check shape and dtype of codes
        if (codes.shape != arr.shape or
                codes.dtype.kind != 'i' or
                codes.itemsize != 1):
            return {"rel_err": 1e9}

        # Exact code match
        if not np.array_equal(codes, ref_codes):
            return {"rel_err": 1e9}

        err = rel_err(arr, deq)
        if err > worst_err:
            worst_err = err

    return {"rel_err": worst_err}
