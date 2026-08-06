import numpy as np


def evaluate_perplexity_curves(seq_lens: list, full_ppl: list, sink_ppl: list, random_ppl: list) -> dict:
    f_arr = np.array(full_ppl, dtype=float)
    s_arr = np.array(sink_ppl, dtype=float)
    r_arr = np.array(random_ppl, dtype=float)

    sink_better_count = int(np.sum(s_arr <= r_arr + 1e-5))
    random_spike_detected = bool(np.max(r_arr) > np.mean(f_arr) * 1.5)

    return {
        "sink_better_count": sink_better_count,
        "random_spike_detected": random_spike_detected,
        "mean_sink_ppl": float(np.mean(s_arr)),
        "mean_random_ppl": float(np.mean(r_arr))
    }
