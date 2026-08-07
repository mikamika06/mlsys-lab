import numpy as np


def generate_workload(seed=42):
    rng = np.random.default_rng(seed)
    workload = []
    for i in range(20):
        prompt_len = int(rng.integers(128, 2048))
        output_len = int(rng.integers(32, 256))
        workload.append({"id": i, "prompt_len": prompt_len, "output_len": output_len})
    return workload


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


def find_pareto_front(results):
    points = [(r["ttft"], r["itl"], r) for r in results]
    points.sort(key=lambda x: x[0])
    pareto = []
    min_itl = float("inf")
    for ttft, itl, r in points:
        if itl < min_itl:
            pareto.append(r)
            min_itl = itl
    return pareto
