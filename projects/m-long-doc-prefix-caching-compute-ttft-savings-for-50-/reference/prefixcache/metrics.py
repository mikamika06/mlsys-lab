def compute_ttft_and_savings(
    sim_results: list[dict],
    model_params: dict,
) -> dict:
    """Calculate prefill compute GFLOPs and TTFT latency metrics with and without prefix caching."""
    gflops_lin = model_params["gflops_per_token_linear"]
    gflops_quad = model_params["gflops_per_token_quad"]
    tflops_cap = model_params["compute_throughput_tflops"]
    overhead_ms = model_params["overhead_ms"]
    gen_ms = model_params["time_per_gen_token_ms"]

    total_gflops_cached = 0.0
    total_gflops_naive = 0.0
    ttft_cached_list = []
    ttft_naive_list = []

    for item in sim_results:
        L = item["total_tokens"]
        U = item["uncached_tokens"]

        gflops_cached = U * gflops_lin + U * L * gflops_quad
        gflops_naive = L * gflops_lin + L * L * gflops_quad

        total_gflops_cached += gflops_cached
        total_gflops_naive += gflops_naive

        latency_cached = (gflops_cached / tflops_cap) + overhead_ms
        latency_naive = (gflops_naive / tflops_cap) + overhead_ms

        ttft_cached_list.append(latency_cached + gen_ms)
        ttft_naive_list.append(latency_naive + gen_ms)

    flops_savings_ratio = (
        1.0 - (total_gflops_cached / total_gflops_naive)
        if total_gflops_naive > 0
        else 0.0
    )
    avg_ttft_cached = sum(ttft_cached_list) / len(ttft_cached_list) if ttft_cached_list else 0.0
    avg_ttft_naive = sum(ttft_naive_list) / len(ttft_naive_list) if ttft_naive_list else 0.0
    avg_ttft_speedup = avg_ttft_naive / avg_ttft_cached if avg_ttft_cached > 0 else 0.0

    speedups_q1_to_q50 = [
        ttft_naive_list[i] / ttft_cached_list[i]
        for i in range(1, len(sim_results))
    ]
    cached_q1_to_q50_avg_ttft_speedup = (
        sum(speedups_q1_to_q50) / len(speedups_q1_to_q50)
        if speedups_q1_to_q50
        else 0.0
    )

    return {
        "total_gflops_cached": float(total_gflops_cached),
        "total_gflops_naive": float(total_gflops_naive),
        "flops_savings_ratio": float(flops_savings_ratio),
        "avg_ttft_cached_ms": float(avg_ttft_cached),
        "avg_ttft_naive_ms": float(avg_ttft_naive),
        "avg_ttft_speedup": float(avg_ttft_speedup),
        "cached_q1_to_q50_avg_ttft_speedup": float(cached_q1_to_q50_avg_ttft_speedup),
    }
