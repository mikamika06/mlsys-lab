import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    res = {"short_exact_match": 0.0}

    try:
        import ref
        from attn.window_sink import compute_window_sink_attention
    except Exception:
        return res

    num_sinks = 2
    window_size = 8
    head_dim = 16
    seq_len = 6

    q, k, v = ref.generate_synthetic_data(seq_len, head_dim, seed=77)

    ref_out = ref.full_causal_attention(q, k, v)

    try:
        user_out = compute_window_sink_attention(
            q, k, v, num_sinks=num_sinks, window_size=window_size
        )
        if np.allclose(user_out, ref_out, atol=1e-6):
            res["short_exact_match"] = 1.0
    except Exception:
        pass

    return res
