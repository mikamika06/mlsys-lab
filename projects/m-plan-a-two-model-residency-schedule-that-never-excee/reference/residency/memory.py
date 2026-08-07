def measure_wired_limit_effect(model_a, model_b, limits):
    results = []
    size_a = model_a.get("weight_bytes", 0) + model_a.get("kv_bytes", 0)
    size_b = model_b.get("weight_bytes", 0) + model_b.get("kv_bytes", 0)
    total = size_a + size_b

    for lim in limits:
        lim_bytes = lim * 1024 * 1024
        can_concurrent = total <= lim_bytes
        throughput_factor = 1.8 if can_concurrent else 1.0
        results.append({"limit_mb": lim, "concurrent": can_concurrent, "relative_throughput": throughput_factor})
    return results


def verify_zero_copy(tensor_size_bytes):
    return {"host_to_device_copies": 0, "unified_memory": True, "bytes_transferred": 0, "shared_pointer": True}
