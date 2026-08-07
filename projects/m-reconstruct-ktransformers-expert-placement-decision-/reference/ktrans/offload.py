def evaluate_offload_latency(num_layers, token_count, gpu_layer_time, cpu_layer_time, pcie_transfer_time, offload_first_n):
    n = max(0, min(num_layers, offload_first_n))
    cpu_layers = n
    gpu_layers = num_layers - n

    layer_time_all_offload = cpu_layer_time * num_layers
    total_offload_all = token_count * layer_time_all_offload

    layer_time_split = (cpu_layers * cpu_layer_time) + (gpu_layers * gpu_layer_time)
    transfer_penalty = pcie_transfer_time if (cpu_layers > 0 and gpu_layers > 0) else 0.0
    total_offload_split = token_count * (layer_time_split + transfer_penalty)

    return {
        "offload_all_latency": float(total_offload_all),
        "offload_split_latency": float(total_offload_split),
        "speedup": float(total_offload_all / total_offload_split) if total_offload_split > 0 else 1.0
    }
