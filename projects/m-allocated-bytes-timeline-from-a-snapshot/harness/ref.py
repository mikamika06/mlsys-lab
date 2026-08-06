import numpy as np


def generate_mock_snapshot(seed=42):
    rng = np.random.RandomState(seed)
    
    traces = []
    active_bytes = 0
    time_step = 0
    
    frames_pool = [
        {"filename": "train.py", "line": 42, "name": "train_step"},
        {"filename": "models/net.py", "line": 105, "name": "forward"},
        {"filename": "utils/logger.py", "line": 18, "name": "log_history"},
        {"filename": "optimizer/adam.py", "line": 88, "name": "step"}
    ]
    
    allocations = {}
    
    for i in range(100):
        time_step += 10
        if rng.rand() > 0.3 or not allocations:
            is_leak = (rng.rand() < 0.15)
            size = int(rng.randint(1, 100)) * 1024 * 1024
            if is_leak:
                frame_idx = 2
            else:
                frame_idx = rng.randint(0, 2)
            
            addr = 0x10000000 + i * 0x10000
            stack = [frames_pool[0], frames_pool[frame_idx]]
            
            allocations[addr] = (size, stack, is_leak)
            active_bytes += size
            
            traces.append({
                "action": "alloc",
                "addr": addr,
                "size": size,
                "time": time_step,
                "frames": stack
            })
        else:
            non_leaks = [a for a, info in allocations.items() if not info[2]]
            if non_leaks:
                target_addr = non_leaks[rng.randint(0, len(non_leaks))]
                size, stack, _ = allocations.pop(target_addr)
                active_bytes -= size
                traces.append({
                    "action": "free",
                    "addr": target_addr,
                    "size": size,
                    "time": time_step,
                    "frames": stack
                })

    return {
        "device_traces": [traces],
        "model_spec": {
            "param_count": 10000000,
            "bytes_per_param": 4,
            "optimizer_multiplier": 3.0
        }
    }


def compute_reference_timeline(snapshot):
    traces = snapshot["device_traces"][0]
    timeline = []
    current_bytes = 0
    peak_bytes = 0
    
    for event in traces:
        if event["action"] == "alloc":
            current_bytes += event["size"]
        elif event["action"] == "free":
            current_bytes -= event["size"]
        
        if current_bytes > peak_bytes:
            peak_bytes = current_bytes
            
        timeline.append({"time": event["time"], "allocated_bytes": current_bytes})
        
    return timeline, peak_bytes


def compute_reference_retaining_frame(snapshot):
    traces = snapshot["device_traces"][0]
    live_allocs = {}
    
    for event in traces:
        if event["action"] == "alloc":
            live_allocs[event["addr"]] = (event["size"], event["frames"])
        elif event["action"] == "free":
            live_allocs.pop(event["addr"], None)
            
    frame_bytes = {}
    for size, frames in live_allocs.values():
        if frames:
            top_frame = f"{frames[-1]['filename']}:{frames[-1]['line']}:{frames[-1]['name']}"
            frame_bytes[top_frame] = frame_bytes.get(top_frame, 0) + size
            
    retaining_frame = max(frame_bytes.items(), key=lambda x: x[1])[0]
    total_retained = frame_bytes[retaining_frame]
    return retaining_frame, total_retained


def compute_reference_footprint(snapshot):
    spec = snapshot["model_spec"]
    params = spec["param_count"]
    bpp = spec["bytes_per_param"]
    opt_mult = spec["optimizer_multiplier"]
    
    theoretical = params * bpp * (1.0 + opt_mult)
    _, peak = compute_reference_timeline(snapshot)
    overhead = peak - theoretical
    return theoretical, overhead
