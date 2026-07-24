import numpy as np
from mlsys.scorers import rel_err

def _reference(K, V):
    # Compute per‑row absolute max scales (float64)
    scales_K = np.max(np.abs(K), axis=1).astype(np.float64)
    scales_V = np.max(np.abs(V), axis=1).astype(np.float64)

    n, dK = K.shape
    _, dV = V.shape

    # Original memory: float32 for all elements
    orig_bytes = (n * dK + n * dV) * 4

    # Quantized memory: int8 per element + float32 scale per row per matrix
    quant_bytes = (n * dK + n * dV) * 1 + n * 4 * 2

    size_ratio = orig_bytes / quant_bytes
    return scales_K, scales_V, float(size_ratio)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        (rng.standard_normal((8, 16)).astype(np.float32),
         rng.standard_normal((8, 12)).astype(np.float32)),
        (rng.standard_normal((4, 5)).astype(np.float32),
         rng.standard_normal((4, 7)).astype(np.float32)),
        (rng.standard_normal((10, 3)).astype(np.float32),
         rng.standard_normal((10, 9)).astype(np.float32)),
        (rng.standard_normal((2, 1)).astype(np.float32),
         rng.standard_normal((2, 1)).astype(np.float32)),
        (rng.standard_normal((6, 20)).astype(np.float32),
         rng.standard_normal((6, 15)).astype(np.float32)),
    ]

    rel_err_scales = 0.0
    size_ratio_err = 0.0

    try:
        for K, V in cases:
            got_K, got_V, got_ratio = sol.compute_scales_and_size(K, V)
            ref_K, ref_V, ref_ratio = _reference(K, V)

            err_k = rel_err(ref_K, np.asarray(got_K))
            err_v = rel_err(ref_V, np.asarray(got_V))
            rel_err_scales = max(rel_err_scales, err_k, err_v)

            size_ratio_err = max(size_ratio_err,
                                 rel_err(np.array([ref_ratio]), np.array([got_ratio])))
    except Exception:
        # Any exception causes maximal error
        return {"rel_err_scales": 1.0, "size_ratio_err": 1.0}

    return {"rel_err_scales": rel_err_scales, "size_ratio_err": size_ratio_err}
