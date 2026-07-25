import numpy as np
from mlsys import scorers

def _reference(Q, K, V, mask=None, causal=False):
    d_k = K.shape[-1]
    scale = 1 / np.sqrt(d_k)
    logits = np.matmul(Q, K.swapaxes(-2, -1)) * scale

    if causal:
        seq_q, seq_k = logits.shape[-2], logits.shape[-1]
        causal_mask = np.triu(np.full((seq_q, seq_k), -np.inf), 1)
        logits += causal_mask

    if mask is not None:
        if mask.dtype == bool:
            logits[~mask] = -np.inf
        else:
            logits += mask

    # stable softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits - max_logits)
    attn_weights = exp / np.sum(exp, axis=-1, keepdims=True)

    output = np.matmul(attn_weights, V)
    return output, attn_weights

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    for _ in range(5):
        batch = rng.integers(1, 4)
        heads = rng.integers(1, 3)
        seq_q = rng.integers(2, 6)
        seq_k = rng.integers(2, 6)
        d_k = rng.integers(4, 8)

        Q = rng.standard_normal((batch, heads, seq_q, d_k))
        K = rng.standard_normal((batch, heads, seq_k, d_k))
        V = rng.standard_normal((batch, heads, seq_k, d_k))

        mask_type = rng.choice(['none', 'causal', 'float', 'bool'])
        causal = False
        mask = None
        if mask_type == 'causal':
            causal = True
        elif mask_type == 'float':
            mask = rng.uniform(-5, 5, (batch, heads, seq_q, seq_k))
        elif mask_type == 'bool':
            mask = rng.choice([True, False], size=(batch, heads, seq_q, seq_k), p=[0.8, 0.2])

        try:
            out, w = sol.scaled_dot_product_attention(Q, K, V, mask=mask, causal=causal)
        except Exception:
            return {"output_error": float("inf"), "weight_error": float("inf")}

        ref_out, ref_w = _reference(Q, K, V, mask=mask, causal=causal)

        out_err = scorers.max_abs_err(ref_out, out)
        w_err   = scorers.max_abs_err(ref_w, w)

        if out_err > 1e-5 or w_err > 1e-5:
            return {"output_error": out_err, "weight_error": w_err}

    return {"output_error": 0.0, "weight_error": 0.0}
