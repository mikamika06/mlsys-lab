"""Benchmark runner for simulated vLLM serving engine."""

import numpy as np


def simulate_request(prompt_tokens, max_tokens, rate):
    base_latency = 0.01 + (prompt_tokens * 0.0001)
    gen_latency = max_tokens * (0.002 + 0.0005 * (rate / 10.0))
    total_latency = base_latency + gen_latency
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": max_tokens,
        "latency": total_latency,
        "ttft": base_latency,
    }


def run_bench_serve(requests, rate, num_workers=1):
    results = []
    if not requests:
        return {"results": [], "duration": 0.0, "rate": rate}

    inter_arrival_times = np.random.default_rng(42).exponential(
        1.0 / rate, len(requests)
    )
    current_time = 0.0
    start_time = 0.0

    for req, iat in zip(requests, inter_arrival_times):
        current_time += iat
        prompt_tokens = req.get("prompt_tokens", 128)
        max_tokens = req.get("max_tokens", 64)
        res = simulate_request(prompt_tokens, max_tokens, rate)
        res["arrival_time"] = current_time
        res["finish_time"] = current_time + res["latency"]
        results.append(res)

    total_duration = (
        max(r["finish_time"] for r in results) - start_time
        if results
        else 0.0
    )
    return {"results": results, "duration": total_duration, "rate": rate}


def run_multi_rate_bench(requests, rates):
    sweep_results = {}
    for r in rates:
        sweep_results[r] = run_bench_serve(requests, r)
    return sweep_results
