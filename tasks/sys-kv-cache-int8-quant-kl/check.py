import numpy as np

from mlsys.scorers import mean_kl, rel_err


def grade(sol, fx) -> dict:
    """Builds random (Q, K, V) attention inputs and compares the
    candidate's int8-KV-cache attention against a plain fp32 (unquantized)
    reference: `mean_kl` on the pre-softmax logits (the attention-weight
    distribution induced by quantized K) and `rel_err` on the attention
    output (affected by quantizing both K and V).
    """
    rng = np.random.default_rng(0)

    worst_kl = 0.0
    worst_rel = 0.0

    for _ in range(4):
        m = int(rng.integers(4, 12))
        n = int(rng.integers(16, 64))
        d = int(rng.integers(8, 32))
        Q = rng.standard_normal((m, d)).astype(np.float32)
        K = (rng.standard_normal((n, d)) * rng.uniform(0.5, 3.0)).astype(np.float32)
        V = (rng.standard_normal((n, d)) * rng.uniform(0.5, 3.0)).astype(np.float32)

        ref_logits = (Q.astype(np.float64) @ K.astype(np.float64).T) / np.sqrt(d)
        z = ref_logits - np.max(ref_logits, axis=-1, keepdims=True)
        w = np.exp(z)
        w = w / np.sum(w, axis=-1, keepdims=True)
        ref_out = w @ V.astype(np.float64)

        try:
            got_logits, got_out = sol.kv_cache_int8_attention(Q, K, V)
            got_logits = np.asarray(got_logits, dtype=np.float64)
            got_out = np.asarray(got_out, dtype=np.float64)
        except Exception:
            return {"mean_kl": float("inf"), "rel_err": float("inf")}

        if got_logits.shape != ref_logits.shape or got_out.shape != ref_out.shape:
            return {"mean_kl": float("inf"), "rel_err": float("inf")}

        worst_kl = max(worst_kl, mean_kl(ref_logits, got_logits))
        worst_rel = max(worst_rel, rel_err(ref_out, got_out))

    return {"mean_kl": float(worst_kl), "rel_err": float(worst_rel)}
