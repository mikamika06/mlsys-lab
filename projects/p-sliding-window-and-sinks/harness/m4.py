import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    res = {"retrieval_accuracy": 0.0, "naive_window_accuracy": 0.0}

    try:
        import ref
        from attn.window_sink import compute_window_sink_attention
    except Exception:
        return res

    seq_len = 100
    head_dim = 16
    needle_pos = 2
    num_sinks = 4
    window_size = 16

    hits_sink = 0
    hits_naive = 0
    num_trials = 10

    for seed in range(num_trials):
        q, k, v, target_val = ref.generate_needle_in_haystack(
            seq_len, head_dim, needle_pos, seed=seed + 500
        )

        try:
            ws_out = compute_window_sink_attention(
                q, k, v, num_sinks=num_sinks, window_size=window_size
            )
            pred_sink = ws_out[-1]
            cos_sim_sink = np.dot(pred_sink, target_val) / (
                np.linalg.norm(pred_sink) * np.linalg.norm(target_val) + 1e-8
            )
            if cos_sim_sink > 0.7:
                hits_sink += 1
        except Exception:
            pass

        naive_out = ref.naive_sliding_window_attention(q, k, v, window_size)
        pred_naive = naive_out[-1]
        cos_sim_naive = np.dot(pred_naive, target_val) / (
            np.linalg.norm(pred_naive) * np.linalg.norm(target_val) + 1e-8
        )
        if cos_sim_naive > 0.7:
            hits_naive += 1

    res["retrieval_accuracy"] = round(hits_sink / num_trials, 2)
    res["naive_window_accuracy"] = round(hits_naive / num_trials, 2)
    return res
