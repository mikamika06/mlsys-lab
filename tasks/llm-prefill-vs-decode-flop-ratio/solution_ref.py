def prefill_vs_decode_flops(d_model: int, n_heads: int, d_ff: int, P: int, T: int) -> dict:
    """Matmul FLOPs of one transformer decoder layer: prefill vs one decode step.

    Counts only matrix-multiply FLOPs (2*m*k*n per A_{m x k} B_{k x n}).

    Prefill (P tokens, self-attention over P positions):
        8*P*d^2  (Q,K,V,O projections)
      + 4*P^2*d  (QK^T and A@V, summed over heads = 2*P^2*d each)
      + 4*P*d*f  (FFN up + down)

    Decode (1 new token attending to T cached keys/values):
        8*d^2    (Q,K,V,O projections for the single new token)
      + 4*T*d    (qK^T over T keys and a@V, summed over heads)
      + 4*d*f    (FFN up + down)
    """
    d, f = d_model, d_ff
    prefill = 8 * P * d * d + 4 * P * P * d + 4 * P * d * f
    decode = 8 * d * d + 4 * T * d + 4 * d * f
    return {"prefill": int(prefill), "decode": int(decode), "ratio": prefill / decode}
