def compute_recompute_time(tokens_to_recompute: int, compute_tflops: float, model_params_b: float) -> float:
    flops = 2.0 * model_params_b * 1e9 * tokens_to_recompute
    throughput = compute_tflops * 1e12
    return flops / throughput


def compute_swap_time(num_blocks: int, block_size_bytes: int, pcie_bandwidth_gbps: float) -> float:
    total_bytes = num_blocks * block_size_bytes * 2.0
    bandwidth = pcie_bandwidth_gbps * 1e9
    return total_bytes / bandwidth


def choose_preemption_mode(tokens_to_recompute: int, num_blocks: int, block_size_bytes: int, compute_tflops: float, model_params_b: float, pcie_bandwidth_gbps: float) -> str:
    t_recompute = compute_recompute_time(tokens_to_recompute, compute_tflops, model_params_b)
    t_swap = compute_swap_time(num_blocks, block_size_bytes, pcie_bandwidth_gbps)
    if t_swap < t_recompute:
        return "swap"
    return "recompute"
