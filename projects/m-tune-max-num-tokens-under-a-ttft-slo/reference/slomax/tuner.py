import numpy as np
from slomax.backend import compute_ttft


def select_max_num_tokens(prompt_lengths, candidate_tokens, slo_ms, backend_type, batching_mode, quantile=0.95):
    best_idx = -1
    best_rate = -1.0

    for idx, candidate in enumerate(candidate_tokens):
        ttfts = [compute_ttft(L, candidate, backend_type, batching_mode) for L in prompt_lengths]
        if any(np.isinf(t) for t in ttfts):
            continue
        p_val = float(np.percentile(ttfts, quantile * 100))
        if p_val <= slo_ms:
            total_tokens = sum(prompt_lengths)
            total_time = sum(ttfts)
            rate = total_tokens / total_time if total_time > 0 else 0.0
            if rate > best_rate:
                best_rate = rate
                best_idx = idx

    return best_idx
