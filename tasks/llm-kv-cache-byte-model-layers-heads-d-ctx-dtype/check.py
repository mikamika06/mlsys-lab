import numpy as np

def _ref(layers, heads, head_dim, seq_len, dtype):
    dt = np.dtype(dtype)
    return int(layers * heads * seq_len * head_dim * 2 * dt.itemsize)

def grade(sol, fx) -> dict:
    cases = [
        (1, 8, 64, 1024, "float32"),
        (12, 16, 128, 2048, "float16"),
        (24, 32, 256, 4096, "int8"),
        (2, 4, 32, 512, "float64"),
    ]
    ok = 1.0
    for layers, heads, d, ctx, dtype in cases:
        try:
            got = sol.kv_cache_bytes(layers, heads, d, ctx, dtype)
        except Exception:
            return {"size_ratio": 0.0}
        expected = _ref(layers, heads, d, ctx, dtype)
        if not isinstance(got, int):
            return {"size_ratio": 0.0}
        ratio = expected / got if got != 0 else 0.0
        if abs(ratio - 1.0) > 1e-9:
            ok = 0.0
            break
    return {"size_ratio": ok}
