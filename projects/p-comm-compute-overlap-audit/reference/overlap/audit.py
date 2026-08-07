import numpy as np

def extract_events(trace_data):
    events = []
    for item in trace_data:
        if "name" in item and "dur" in item:
            events.append({
                "name": item["name"],
                "start": item.get("start", 0),
                "dur": item["dur"],
                "type": item.get("type", "compute")
            })
    return events

def calculate_traffic(model_config, world_size):
    hidden_size = model_config.get("hidden_size", 4096)
    num_layers = model_config.get("num_layers", 32)
    dtype_bytes = model_config.get("dtype_bytes", 2)
    params_per_layer = 12 * (hidden_size ** 2)
    total_params = num_layers * params_per_layer
    bytes_per_param = dtype_bytes
    total_bytes = total_params * bytes_per_param
    ring_factor = 2.0 * (world_size - 1) / world_size
    traffic_per_step = total_bytes * ring_factor
    return int(traffic_per_step)

def find_barriers(events):
    barriers = []
    sorted_events = sorted(events, key=lambda x: x["start"])
    for i in range(len(sorted_events) - 1):
        curr = sorted_events[i]
        nxt = sorted_events[i+1]
        if curr["type"] == "compute" and nxt["type"] == "comm":
            if nxt["start"] >= (curr["start"] + curr["dur"]):
                barriers.append((curr, nxt))
    return barriers

def optimize_buckets(tensors, target_size):
    buckets = []
    current_bucket = []
    current_size = 0
    for t in tensors:
        size = t.get("size", 1024)
        if current_size + size > target_size and current_bucket:
            buckets.append(current_bucket)
            current_bucket = [t]
            current_size = size
        else:
            current_bucket.append(t)
            current_size += size
    if current_bucket:
        buckets.append(current_bucket)
    return buckets

def compute_overlap_ratio(events):
    comm_events = [e for e in events if e["type"] == "comm"]
    compute_events = [e for e in events if e["type"] == "compute"]
    if not comm_events:
        return 0.0
    total_comm_dur = sum(e["dur"] for e in comm_events)
    overlapped_dur = 0
    for ce in comm_events:
        c_start = ce["start"]
        c_end = c_start + ce["dur"]
        overlap = False
        for comp in compute_events:
            comp_start = comp["start"]
            comp_end = comp_start + comp["dur"]
            if not (c_end <= comp_start or c_start >= comp_end):
                overlap = True
                break
        if overlap:
            overlapped_dur += ce["dur"]
    unoverlapped_ratio = 1.0 - (overlapped_dur / total_comm_dur) if total_comm_dur > 0 else 0.0
    return float(unoverlapped_ratio)

def forecast_scaling(base_time, comm_time, world_size):
    effective_comm = comm_time * (1.0 + 0.1 * np.log2(world_size))
    total_time = base_time + effective_comm / world_size
    return float(total_time)
