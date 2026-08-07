import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    res = {"budget_met": 0.0, "quality_retained": 0.0}

    try:
        import ref
        from attn.window_sink import StreamingAttentionRunner
    except Exception:
        return res

    seq_len = 150
    head_dim = 16
    needle_pos = 1
    num_sinks = 2
    window_size = 20
    max_allowed_cache = num_sinks + window_size

    q, k, v, target_val = ref.generate_needle_in_haystack(
        seq_len, head_dim, needle_pos, seed=999
    )

    try:
        runner = StreamingAttentionRunner(num_sinks, window_size, head_dim)
        peak_len = 0
        last_out = None
        for i in range(seq_len):
            last_out = runner.step(q[i], k[i], v[i])
            peak_len = max(peak_len, runner.cache.current_seq_len)

        if peak_len <= max_allowed_cache:
            res["budget_met"] = 1.0

        if last_out is not None:
            pred = last_out[0]
            cos_sim = np.dot(pred, target_val) / (
                np.linalg.norm(pred) * np.linalg.norm(target_val) + 1e-8
            )
            if cos_sim > 0.7:
                res["quality_retained"] = 1.0
    except Exception:
        pass

    return res
