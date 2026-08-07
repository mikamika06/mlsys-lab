import numpy as np

def compute_theoretical_volume(model_config, world_size):
    hidden_size = model_config.get("hidden_size", 4096)
    num_layers = model_config.get("num_layers", 32)
    bytes_per_param = model_config.get("bytes_per_param", 2)
    params_per_layer = 12 * (hidden_size ** 2)
    total_params = num_layers * params_per_layer
    total_bytes = total_params * bytes_per_param
    ring_factor = 2.0 * (world_size - 1) / world_size
    return int(total_bytes * ring_factor)

def find_barriers(events):
    barriers = []
    for i, ev in enumerate(events):
        if ev.get("sync_point", False) or (ev.get("name") == "all_reduce" and not ev.get("async", False)):
            barriers.append(i)
    return barriers

def optimize_buckets(layers, target_bucket_size):
    buckets = []
    current_bucket = []
    current_size = 0
    for layer in layers:
        size = layer.get("size", 1024)
        if current_size + size > target_bucket_size and current_bucket:
            buckets.append(current_bucket)
            current_bucket = [layer["id"]]
            current_size = size
        else:
            current_bucket.append(layer["id"])
            current_size += size
    if current_bucket:
        buckets.append(current_bucket)
    return buckets

def measure_unoverlapped_ratio(timeline):
    total_compute = sum(ev["dur"] for ev in timeline if ev.get("cat") == "compute")
    total_comm = sum(ev["dur"] for ev in timeline if ev.get("cat") == "comm")
    overlapped = sum(min(ev.get("compute_dur", 0), ev.get("comm_dur", 0)) for ev in timeline if "overlapped" in ev)
    unoverlapped_comm = max(0.0, total_comm - overlapped)
    if total_comm == 0:
        return 0.0
    return float(unoverlapped_comm / total_comm)

def predict_scaling(base_time, comm_volume, world_size):
    bandwidth_gbps = 100.0
    comm_time = comm_volume / (bandwidth_gbps * 1e9)
    estimated_time = base_time + comm_time * np.log2(world_size)
    return float(estimated_time)
