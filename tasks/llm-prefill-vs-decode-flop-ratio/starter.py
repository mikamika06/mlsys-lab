def prefill_vs_decode_flops(d_model: int, n_heads: int, d_ff: int, P: int, T: int) -> dict:
    """Return {"prefill": int, "decode": int, "ratio": float} of matmul FLOPs for
    one transformer decoder layer: a P-token prefill vs a single decode step that
    attends to T cached keys. Count only matmul FLOPs (2*m*k*n per matmul)."""
    raise NotImplementedError("your code here")
