import numpy as np


def select_max_batched_tokens(workloads, token_candidates, itl_p99_limit_ms=40.0):
    best_tokens = None
    max_goodput = -1.0
    for tokens in sorted(token_candidates):
        goodputs = []
        itls = []
        for wl in workloads:
            sim_goodput, sim_itl_p99 = np.array(wl(tokens), dtype=float)
            goodputs.append(sim_goodput)
            itls.append(sim_itl_p99)
        mean_itl_p99 = float(np.percentile(itls, 99))
        avg_goodput = float(np.mean(goodputs))
        if mean_itl_p99 <= itl_p99_limit_ms:
            if avg_goodput > max_goodput:
                max_goodput = avg_goodput
                best_tokens = tokens
    if best_tokens is None:
        best_tokens = int(token_candidates[0])
    return int(best_tokens)
