import numpy as np
from mlsys.scorers import max_abs_err

def _oracle(Q, K, V, causal):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if causal:
        seq_len = scores.shape[1]
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        scores += mask
    # softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return attn_weights @ V

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0
    for batch in [1, 2]:
        for seq_len in [4, 8]:
            d_k = 16
            d_v = 16
            Q = rng.standard_normal((batch, seq_len, d_k), dtype=np.float64)
            K = rng.standard_normal((batch, seq_len, d_k), dtype=np.float64)
            V = rng.standard_normal((batch, seq_len, d_v), dtype=np.float64)

            try:
                got = sol.scaled_dot_product_attention(Q, K, V, causal=False)
                ref = _oracle(Q, K, V, causal=False)
                err = max_abs_err(ref, got)
                if err > max_err:
                    max_err = err
            except Exception:
                return {"max_abs_err": float("inf")}

            try:
                got_c = sol.scaled_dot_product_attention(Q, K, V, causal=True)
                ref_c = _oracle(Q, K, V, causal=True)
                err_c = max_abs_err(ref_c, got_c)
                if err_c > max_err:
                    max_err = err_c
            except Exception:
                return {"max_abs_err": float("inf")}

    return {"max_abs_err": max_err}
