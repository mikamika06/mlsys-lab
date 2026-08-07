def estimate_memory_and_time(tensor_total_bytes, world_size, bandwidth_gbps):
    sharded_memory = tensor_total_bytes // world_size
    transfer_time = (tensor_total_bytes / (bandwidth_gbps * 1024 * 1024 * 1024))
    return int(sharded_memory), float(transfer_time)
