"""Fallback cost measurement."""

BACKEND_EFFICIENCY = {
    "FA3_HOPPER": 0.75,
    "FA2_CUDA": 0.60,
    "FA2_TRITON": 0.50,
    "MATH_FALLBACK": 0.15,
}


def _estimate_tflops(machine_config, dtype):
    sm_count = machine_config.get("sm_count", 108)
    clock_ghz = machine_config.get("clock_ghz", 1.5)
    cc = machine_config.get("compute_capability", (8, 0))

    if cc >= (9, 0):
        base_flops = 1000.0 if dtype in ("float16", "bfloat16", "float8") else 200.0
    elif cc >= (8, 0):
        base_flops = 300.0 if dtype in ("float16", "bfloat16") else 100.0
    else:
        base_flops = 130.0 if dtype in ("float16", "bfloat16") else 50.0

    return (sm_count / 108.0) * (clock_ghz / 1.5) * base_flops


def measure_fallback_cost(target_backend, fallback_backend, machine_config, input_spec):
    batch = input_spec.get("batch_size", 1)
    seq_len = input_spec.get("seq_len", 2048)
    num_heads = input_spec.get("num_heads", 32)
    head_dim = input_spec.get("head_dim", 128)
    dtype = input_spec.get("dtype", "float16")

    bytes_per_elem = 1 if dtype == "float8" else 2
    total_flops = 4.0 * batch * num_heads * (seq_len ** 2) * head_dim

    target_eff = BACKEND_EFFICIENCY.get(target_backend, 0.50)
    fallback_eff = BACKEND_EFFICIENCY.get(fallback_backend, 0.15)

    peak_tflops = _estimate_tflops(machine_config, dtype)
    peak_flops = peak_tflops * 1e12

    target_time_sec = total_flops / (peak_flops * target_eff)
    fallback_time_sec = total_flops / (peak_flops * fallback_eff)

    qkv_bytes = 3 * batch * seq_len * num_heads * head_dim * bytes_per_elem
    out_bytes = batch * seq_len * num_heads * head_dim * bytes_per_elem

    target_mem_bytes = qkv_bytes + out_bytes
    if fallback_backend == "MATH_FALLBACK":
        attn_matrix_bytes = batch * num_heads * seq_len * seq_len * 4
        fallback_mem_bytes = target_mem_bytes + attn_matrix_bytes
    else:
        fallback_mem_bytes = target_mem_bytes

    latency_penalty_ratio = fallback_time_sec / target_time_sec
    memory_overhead_bytes = fallback_mem_bytes - target_mem_bytes

    return {
        "target_backend": target_backend,
        "fallback_backend": fallback_backend,
        "target_latency_ms": target_time_sec * 1000.0,
        "fallback_latency_ms": fallback_time_sec * 1000.0,
        "latency_penalty_ratio": latency_penalty_ratio,
        "memory_overhead_bytes": memory_overhead_bytes,
        "total_flops": total_flops,
    }
