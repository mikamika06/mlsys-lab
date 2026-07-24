import numpy as np

def _reference(num_layers, batch_size, num_heads, seq_len, head_dim, dtype):
    """Compute the expected peak KV bytes for both strategies."""
    b = dtype.itemsize
    layer_kv = 2 * batch_size * num_heads * seq_len * head_dim * b
    peak_off = 2 * layer_kv
    peak_full = num_layers * layer_kv
    return (float(peak_off), float(peak_full))

def grade(sol, fx) -> dict:
    # Generate deterministic random test cases
    rng = np.random.default_rng(seed=0)
    ok = 1.0
    for _ in range(5):
        num_layers = int(rng.integers(2, 20))
        batch_size = int(rng.integers(1, 64))
        num_heads   = int(rng.integers(1, 32))
        seq_len     = int(rng.integers(16, 512))
        head_dim    = int(rng.integers(8, 256))
        dtype       = np.dtype(np.float32) if rng.random() < 0.5 else np.dtype(np.float64)
        try:
            got = sol.kv_peak_bytes(num_layers, batch_size, num_heads,
                                    seq_len, head_dim, dtype)
            ref = _reference(num_layers, batch_size, num_heads,
                             seq_len, head_dim, dtype)
        except Exception:
            ok = 0.0
            break
        if not isinstance(got, tuple) or len(got) != 2:
            ok = 0.0
            break
        # Allow tiny floating point differences
        if not (np.isclose(got[0], ref[0], atol=1e-6) and np.isclose(got[1], ref[1], atol=1e-6)):
            ok = 0.0
            break
    return {"exact_match": ok}
