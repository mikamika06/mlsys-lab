from streamllm.cache import compute_perplexity


def find_optimal_sink(seq_len, window_size, candidates):
    best_s = candidates[0]
    best_val = float("inf")
    for s in candidates:
        val = compute_perplexity(seq_len, s, window_size, "sink_window")
        if val < best_val:
            best_val = val
            best_s = s
    return int(best_s)
