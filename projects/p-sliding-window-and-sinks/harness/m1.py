import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    res = {"api_ok": 0.0, "attention_shape_ok": 0.0, "sinks_preserved": 0.0}

    try:
        from attn.cache import WindowSinkKVCache
        from attn.window_sink import compute_window_sink_attention
    except Exception:
        return res

    num_sinks = 2
    window_size = 4
    head_dim = 8
    seq_len = 10

    try:
        cache = WindowSinkKVCache(num_sinks, window_size, head_dim)
        if cache.capacity != num_sinks + window_size:
            return res
        res["api_ok"] = 1.0
    except Exception:
        return res

    q = np.random.randn(seq_len, head_dim)
    k = np.random.randn(seq_len, head_dim)
    v = np.random.randn(seq_len, head_dim)

    try:
        out = compute_window_sink_attention(q, k, v, num_sinks, window_size)
        if out.shape == (seq_len, head_dim) and not np.isnan(out).any():
            res["attention_shape_ok"] = 1.0
    except Exception:
        return res

    try:
        cache.append(k, v)
        stored_k = cache.get_keys()
        if stored_k.shape[0] == (num_sinks + window_size):
            if np.allclose(stored_k[:num_sinks], k[:num_sinks]):
                res["sinks_preserved"] = 1.0
    except Exception:
        pass

    return res
