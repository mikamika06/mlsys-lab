import numpy as np


def simulate_server(workload, max_num_batched_tokens):
    ttfts = []
    itls = []
    for req in workload:
        p = req["prompt_len"]
        o = req["output_len"]
        ttft = 10.0 + (p / float(max_num_batched_tokens)) * 5.0 + max_num_batched_tokens * 0.001
        itl = 2.0 + 500.0 / float(max_num_batched_tokens)
        ttfts.append(ttft)
        itls.append(itl)
    return float(np.mean(ttfts)), float(np.mean(itls))


def run_sweep(workload, budgets):
    results = []
    for b in budgets:
        ttft, itl = simulate_server(workload, b)
        results.append({"budget": b, "ttft": ttft, "itl": itl})
    return results
