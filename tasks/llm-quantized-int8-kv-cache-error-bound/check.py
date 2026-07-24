import numpy as np
from mlsys.scorers import mean_kl

def _reference_attention(q, k):
    """Compute softmax attention distribution from full‑precision keys."""
    d_k = q.shape[-1]
    scores = q @ k.T / np.sqrt(d_k)
    # stable softmax
    maxes = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - maxes)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

def grade(sol, fx) -> dict:
    try:
        kv_cache_quantize = getattr(sol, "kv_cache_quantize")
    except AttributeError:
        return {"mean_kl": float("inf")}

    # generate several random test cases
    rng = np.random.default_rng(42)
    kl_values = []

    for _ in range(5):
        n, d = rng.integers(4, 12), rng.integers(8, 16)
        q = rng.standard_normal((n, d)).astype(np.float32)
        keys_fp16 = rng.standard_normal((n, d)).astype(np.float16)
        values_fp16 = rng.standard_normal((n, d)).astype(np.float16)

        # reference distribution
        ref_dist = _reference_attention(q, keys_fp16.astype(np.float32))

        try:
            keys_int8, scales = kv_cache_quantize(keys_fp16, values_fp16)
        except Exception:
            return {"mean_kl": float("inf")}

        if keys_int8.dtype != np.int8 or scales.ndim != 1:
            return {"mean_kl": float("inf")}

        # reconstruct approximate keys
        approx_keys = keys_int8.astype(np.float32) * scales[:, None]
        approx_dist = _reference_attention(q, approx_keys)

        kl = mean_kl(ref_dist, approx_dist)
        kl_values.append(kl)

    mean_kl_value = float(np.mean(kl_values))
    return {"mean_kl": mean_kl_value}
