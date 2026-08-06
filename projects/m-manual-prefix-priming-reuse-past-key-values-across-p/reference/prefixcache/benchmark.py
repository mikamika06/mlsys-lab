import math


def benchmark_cache_implementations(
    num_layers,
    num_kv_heads,
    head_dim,
    batch_size,
    max_seq_len,
    current_seq_len,
    dtype_bytes=2,
    quant_bits=8,
    offload_gpu_fraction=0.2,
):
    base_elements_per_token = 2 * num_layers * batch_size * num_kv_heads * head_dim
    static_bytes = base_elements_per_token * max_seq_len * dtype_bytes
    static_peak = static_bytes

    dynamic_bytes = base_elements_per_token * current_seq_len * dtype_bytes
    dynamic_peak = static_bytes

    offloaded_bytes = int(math.ceil(static_bytes * offload_gpu_fraction))
    offloaded_peak = offloaded_bytes

    quant_elem_bytes = quant_bits / 8.0
    quantized_bytes = int(
        math.ceil(base_elements_per_token * current_seq_len * quant_elem_bytes)
    )
    quantized_peak = int(
        math.ceil(base_elements_per_token * max_seq_len * quant_elem_bytes)
    )

    def calc_savings(alloc):
        if static_bytes == 0:
            return 0.0
        return float(1.0 - (alloc / static_bytes))

    return {
        "dynamic": {
            "allocated_bytes": dynamic_bytes,
            "peak_bytes": dynamic_peak,
            "memory_savings_vs_static": calc_savings(dynamic_bytes),
            "supports_dynamic_growth": True,
        },
        "static": {
            "allocated_bytes": static_bytes,
            "peak_bytes": static_peak,
            "memory_savings_vs_static": 0.0,
            "supports_dynamic_growth": False,
        },
        "offloaded": {
            "allocated_bytes": offloaded_bytes,
            "peak_bytes": offloaded_peak,
            "memory_savings_vs_static": calc_savings(offloaded_bytes),
            "supports_dynamic_growth": False,
        },
        "quantized": {
            "allocated_bytes": quantized_bytes,
            "peak_bytes": quantized_peak,
            "memory_savings_vs_static": calc_savings(quantized_bytes),
            "supports_dynamic_growth": True,
        },
    }
