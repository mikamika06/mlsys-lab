import numpy as np

def _reference(
    x, wq, wk, wv,
    rope_freqs,
    kv_cache_k_ref, kv_cache_v_ref,
    cache_pos
):
    # Linear projections
    q = x @ wq
    k = x @ wk
    v = x @ wv

    seq_len = x.shape[1]
    d_model = x.shape[2]

    # RoPE angles
    pos = np.arange(seq_len, dtype=np.float64)[:, None]          # (seq_len,1)
    freq = rope_freqs[None, :]                                   # (1,d//2)
    angle = pos * freq                                           # (seq_len,d//2)
    cos = np.cos(angle)                                          # (seq_len,d//2)
    sin = np.sin(angle)                                          # (seq_len,d//2)

    # Expand to match batch and seq dimensions
    cos_tiled = cos[None, :, :]                                   # (1,seq_len,d//2)
    sin_tiled = sin[None, :, :]

    # Apply RoPE to K
    k_even = k[..., ::2]
    k_odd  = k[..., 1::2]
    k_rot = np.empty_like(k)
    k_rot[..., ::2] = k_even * cos_tiled - k_odd * sin_tiled
    k_rot[..., 1::2] = k_even * sin_tiled + k_odd * cos_tiled

    # Apply RoPE to V
    v_even = v[..., ::2]
    v_odd  = v[..., 1::2]
    v_rot = np.empty_like(v)
    v_rot[..., ::2] = v_even * cos_tiled - v_odd * sin_tiled
    v_rot[..., 1::2] = v_even * sin_tiled + v_odd * cos_tiled

    # Write into cache in place
    kv_cache_k_ref[:, cache_pos:cache_pos+seq_len, :] = k_rot
    kv_cache_v_ref[:, cache_pos:cache_pos+seq_len, :] = v_rot

    return q, k_rot, v_rot


def max_abs_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


def grade(sol, fx) -> dict:
    # Generate several random test cases
    rng = np.random.default_rng(42)
    max_error = 0.0

    for _ in range(5):
        batch = rng.integers(1, 4)
        seq_len = rng.integers(1, 6)
        d_model = rng.choice([8, 12, 16])  # even dimensions
        max_seq_len = seq_len + rng.integers(3, 7)

        x = rng.standard_normal((batch, seq_len, d_model))
        wq = rng.standard_normal((d_model, d_model))
        wk = rng.standard_normal((d_model, d_model))
        wv = rng.standard_normal((d_model, d_model))

        rope_freqs = 1.0 / (10000 ** (np.arange(d_model//2) / d_model))

        kv_cache_k = np.zeros((batch, max_seq_len, d_model), dtype=np.float64)
        kv_cache_v = np.zeros((batch, max_seq_len, d_model), dtype=np.float64)

        cache_pos = rng.integers(0, max_seq_len - seq_len + 1)

        # Copies for reference
        kv_cache_k_ref = kv_cache_k.copy()
        kv_cache_v_ref = kv_cache_v.copy()

        try:
            q_sol, k_rot_sol, v_rot_sol = sol.fused_qkv_rope_kv_cache_write(
                x, wq, wk, wv,
                rope_freqs,
                kv_cache_k, kv_cache_v,
                cache_pos
            )
        except Exception as e:
            return {"max_abs_err": float("inf")}

        q_ref, k_rot_ref, v_rot_ref = _reference(
            x, wq, wk, wv,
            rope_freqs,
            kv_cache_k_ref, kv_cache_v_ref,
            cache_pos
        )

        # Compare outputs
        err_q = max_abs_err(q_sol, q_ref)
        err_k = max_abs_err(k_rot_sol, k_rot_ref)
        err_v = max_abs_err(v_rot_sol, v_rot_ref)

        # Compare caches
        err_cache_k = max_abs_err(kv_cache_k[:, cache_pos:cache_pos+seq_len, :],
                                  kv_cache_k_ref[:, cache_pos:cache_pos+seq_len, :])
        err_cache_v = max_abs_err(kv_cache_v[:, cache_pos:cache_pos+seq_len, :],
                                  kv_cache_v_ref[:, cache_pos:cache_pos+seq_len, :])

        case_max = max(err_q, err_k, err_v, err_cache_k, err_cache_v)
        if case_max > max_error:
            max_error = case_max

    return {"max_abs_err": max_error}
