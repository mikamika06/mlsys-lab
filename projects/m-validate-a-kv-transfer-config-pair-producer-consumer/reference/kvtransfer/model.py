def estimate_pipelined_transfer_time(model_cfg: dict, net_cfg: dict) -> dict:
    """Estimate layer-wise pipelined transfer time and speedup."""
    num_layers = model_cfg["num_layers"]
    compute_ms = model_cfg["layer_compute_ms"]
    layer_bytes = model_cfg["layer_kv_bytes"]

    bandwidth_gbps = net_cfg["bandwidth_gbps"]
    latency_ms = net_cfg.get("latency_ms", 0.0)

    bytes_per_ms = (bandwidth_gbps * 1e9 / 8.0) / 1000.0
    transfer_ms_per_layer = (layer_bytes / bytes_per_ms) + latency_ms

    sequential_time = num_layers * (compute_ms + transfer_ms_per_layer)

    pipelined_time = compute_ms
    for i in range(num_layers):
        if i == num_layers - 1:
            pipelined_time += transfer_ms_per_layer
        else:
            pipelined_time += max(compute_ms, transfer_ms_per_layer)

    speedup = sequential_time / pipelined_time if pipelined_time > 0 else 1.0

    return {
        "sequential_ms": float(sequential_time),
        "pipelined_ms": float(pipelined_time),
        "transfer_per_layer_ms": float(transfer_ms_per_layer),
        "speedup": float(speedup),
    }
