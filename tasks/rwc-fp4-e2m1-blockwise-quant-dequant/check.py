import numpy as np

def _ref_quant_dequant(x: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation of FP4 e2m1 blockwise quant/dequant."""
    flat = x.ravel()
    n = flat.size
    codes = np.empty_like(flat, dtype=np.int8)
    deq = np.empty_like(flat, dtype=np.float64)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = flat[start:end]
        alpha = np.max(np.abs(block))
        # avoid division by zero
        s = alpha / 7.0 if alpha != 0 else 1.0
        q = np.round(block / s)
        q = np.clip(q, -8, 7).astype(np.int8)
        codes[start:end] = q
        deq[start:end] = q.astype(np.float64) * s

    return codes.reshape(x.shape), deq.reshape(x.shape)

def grade(sol, fx) -> dict:
    """Grade the student's implementation."""
    # Generate random test cases
    rng = np.random.default_rng(12345)
    shapes = [(10,), (50,), (200,), (3, 7, 11)]
    block_sizes = [8, 16, 32, 64]
    ok = 1.0

    for shape in shapes:
        for bs in block_sizes:
            x = rng.standard_normal(shape).astype(np.float64)
            try:
                codes_sol, deq_sol = sol.fp4_quant_dequant(x, block_size=bs)
            except Exception:
                return {"exact_match": 0.0}
            # Reference
            codes_ref, deq_ref = _ref_quant_dequant(x, bs)

            if not np.array_equal(codes_sol, codes_ref):
                ok = 0.0
                break
            if not np.allclose(deq_sol, deq_ref, atol=1e-12, rtol=0):
                ok = 0.0
                break
        if ok == 0.0:
            break

    return {"exact_match": ok}
