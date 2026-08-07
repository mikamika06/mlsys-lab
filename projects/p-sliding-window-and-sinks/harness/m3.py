import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    res = {"memory_bounded": 0.0, "unbounded_ratio": 0.0}

    try:
        import ref
        from attn.window_sink import StreamingAttentionRunner
    except Exception:
        return res

    num_sinks = 4
    window_size = 16
    head_dim = 32
    long_seq_len = 200

    q, k, v = ref.generate_synthetic_data(long_seq_len, head_dim, seed=101)

    try:
        runner = StreamingAttentionRunner(num_sinks, window_size, head_dim)
        max_cache_len = 0
        for i in range(long_seq_len):
            _ = runner.step(q[i], k[i], v[i])
            max_cache_len = max(max_cache_len, runner.cache.current_seq_len)

        bounded_cap = num_sinks + window_size
        if max_cache_len <= bounded_cap:
            res["memory_bounded"] = 1.0

        ratio = float(long_seq_len) / float(max_cache_len)
        res["unbounded_ratio"] = round(ratio, 2)
    except Exception:
        pass

    return res
