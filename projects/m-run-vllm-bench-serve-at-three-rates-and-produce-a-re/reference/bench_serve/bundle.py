"""Result bundle generation and formatting."""

import numpy as np


def calculate_metrics(results, total_duration):
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


def create_result_bundle(model_name, rates_data):
    bundle = {
        "model": model_name,
        "rates": {},
        "summary": {"max_throughput": 0.0, "tested_rates": []},
    }

    max_tp = 0.0
    for rate, data in rates_data.items():
        metrics = calculate_metrics(data["results"], data["duration"])
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
