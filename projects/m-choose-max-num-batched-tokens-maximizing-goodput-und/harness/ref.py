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


def generate_workloads():
    rng = np.random.default_rng(42)
    workloads = []
    for _ in range(5):
        base_goodput = rng.uniform(100.0, 500.0)
        base_itl = rng.uniform(20.0, 30.0)
        def make_wl(bg, bi, r):
            def wl(tokens):
                g = bg + float(tokens) * 0.05 + r.normal(0, 5)
                i = bi + (float(tokens) / 500.0) * 8.0 + abs(r.normal(0, 2))
                return [max(10.0, g), max(5.0, i)]
            return wl
        workloads.append(make_wl(base_goodput, base_itl, rng))
    return workloads


CONFIGS = [
    {"max_num_batched_tokens": 2048, "enable_chunked_prefill": True},
    {"max_num_batched_tokens": 4096, "enable_chunked_prefill": True},
    {"max_num_batched_tokens": -1, "enable_chunked_prefill": True},
]
