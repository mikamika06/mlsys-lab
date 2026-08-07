"""Reference data and oracle implementations for grading harness."""

import numpy as np

REQUEST_DATA = [
    {"prompt_tokens": 128, "max_tokens": 64},
    {"prompt_tokens": 256, "max_tokens": 128},
    {"prompt_tokens": 64, "max_tokens": 32},
    {"prompt_tokens": 512, "max_tokens": 256},
    {"prompt_tokens": 128, "max_tokens": 64},
] * 10

RATES = [1.0, 5.0, 10.0]


def oracle_simulate_request(prompt_tokens, max_tokens, rate):
    base_latency = 0.01 + (prompt_tokens * 0.0001)
    gen_latency = max_tokens * (0.002 + 0.0005 * (rate / 10.0))
    total_latency = base_latency + gen_latency
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": max_tokens,
        "latency": total_latency,
        "ttft": base_latency,
    }


def oracle_run_bench_serve(requests, rate):
    results = []
    if not requests:
        return {"results": [], "duration": 0.0, "rate": rate}

    inter_arrival_times = np.random.default_rng(42).exponential(
        1.0 / rate, len(requests)
    )
    current_time = 0.0

    for req, iat in zip(requests, inter_arrival_times):
        current_time += iat
        prompt_tokens = req.get("prompt_tokens", 128)
        max_tokens = req.get("max_tokens", 64)
        res = oracle_simulate_request(prompt_tokens, max_tokens, rate)
        res["arrival_time"] = current_time
        res["finish_time"] = current_time + res["latency"]
        results.append(res)

    total_duration = max(r["finish_time"] for r in results) if results else 0.0
    return {"results": results, "duration": total_duration, "rate": rate}


def oracle_run_multi_rate_bench(requests, rates):
    return {r: oracle_run_bench_serve(requests, r) for r in rates}


def oracle_calculate_metrics(results, total_duration):
    if not results or total_duration <= 0:
        return {
            "completed_requests": 0,
            "total_throughput_tok_s": 0.0,
            "mean_latency": 0.0,
            "p99_latency": 0.0,
            "mean_ttft": 0.0,
        }

    total_tokens = sum(
        r["prompt_tokens"] + r["completion_tokens"] for r in results
    )
    latencies = [r["latency"] for r in results]
    ttfts = [r["ttft"] for r in results]

    return {
        "completed_requests": len(results),
        "total_throughput_tok_s": float(total_tokens / total_duration),
        "mean_latency": float(np.mean(latencies)),
        "p99_latency": float(np.percentile(latencies, 99)),
        "mean_ttft": float(np.mean(ttfts)),
    }


def oracle_create_result_bundle(model_name, rates_data):
    bundle = {
        "model": model_name,
        "rates": {},
        "summary": {"max_throughput": 0.0, "tested_rates": []},
    }

    max_tp = 0.0
    for rate, data in rates_data.items():
        metrics = oracle_calculate_metrics(data["results"], data["duration"])
        bundle["rates"][str(rate)] = {
            "metrics": metrics,
            "raw_results": data["results"],
            "duration": data["duration"],
        }
        bundle["summary"]["tested_rates"].append(rate)
        if metrics["total_throughput_tok_s"] > max_tp:
            max_tp = metrics["total_throughput_tok_s"]

    bundle["summary"]["max_throughput"] = max_tp
    return bundle
