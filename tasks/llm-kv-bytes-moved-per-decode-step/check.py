import numpy as np

def _reference(n_layers, n_heads, d_head, ctx_len, dtype_bytes):
    """Compute the expected bytes using NumPy's dtype oracle for the byte width."""
    dtype_map = {1: np.int8, 2: np.float16, 4: np.float32, 8: np.float64}
    np_dtype = dtype_map[dtype_bytes]
    # NumPy oracle: this is the real, platform-verified byte count per element
    actual_b = np.dtype(np_dtype).itemsize
    return n_layers * 2 * n_heads * d_head * ctx_len * actual_b

def grade(sol, fx) -> dict:
    cases = [
        # (n_layers, n_heads, d_head, ctx_len, dtype_bytes)
        (12, 12, 64, 1024, 2),       # GPT-2 small, fp16
        (32, 32, 128, 2048, 2),      # LLaMA-7B, fp16
        (6, 16, 64, 512, 4),         # small model, fp32
        (24, 16, 128, 1024, 1),      # int8 quantised
        (80, 64, 128, 8192, 2),      # large model, long ctx, fp16
    ]

    ok = 1.0
    for n_layers, n_heads, d_head, ctx_len, dtype_bytes in cases:
        ref = _reference(n_layers, n_heads, d_head, ctx_len, dtype_bytes)
        try:
            got = sol.kv_bytes_per_decode(
                n_layers, n_heads, d_head, ctx_len, dtype_bytes
            )
            got = int(got)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break

    return {"bytes_formula_correct": ok}
