import numpy as np

def grade(sol, fx) -> dict:
    cases = [
        # layers, heads, d_kv, seq_len, dtype, batch
        (1, 8, 64, 128, np.float32, 4),
        (2, 12, 32, 256, np.int16, 8),
        (3, 6, 128, 512, np.float64, 2),
        (5, 10, 64, 1024, np.uint8, 1),
    ]
    ratios = []
    for layers, heads, d_kv, seq_len, dtype, batch in cases:
        try:
            got = sol.kv_cache_bytes(layers, heads, d_kv, seq_len, dtype, batch)
        except Exception:
            return {"size_ratio": 0.0}
        ref = int(
            layers
            * heads
            * 2
            * batch
            * seq_len
            * d_kv
            * np.dtype(dtype).itemsize
        )
        # Compute numeric ratio; must be exactly 1.0 for correct implementation.
        if ref == 0:
            ratio = 1.0
        else:
            ratio = float(got) / float(ref)
        ratios.append(ratio)
    min_ratio = min(ratios) if ratios else 0.0
    return {"size_ratio": float(min_ratio)}
