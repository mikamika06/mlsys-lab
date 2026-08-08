def classify_ncu(metrics):
    res = {}
    for m in metrics:
        if m["sm_pct"] >= 80.0:
            res[m["name"]] = "compute_bound"
        elif m["mem_pct"] >= 80.0:
            res[m["name"]] = "memory_bound"
        elif m["warps_pct"] < 60.0:
            res[m["name"]] = "occupancy_bound"
        else:
            res[m["name"]] = "latency_bound"
    return res

def analyze_proton(events):
    if not events:
        return {}
    stack = []
    exclusive_time = {}
    last_time = None
    for ev in events:
        t = ev["time"]
        if last_time is not None and stack:
            top = stack[-1]
            exclusive_time[top] = exclusive_time.get(top, 0.0) + (t - last_time)
        if ev["type"] == "enter":
            stack.append(ev["region"])
        else:
            stack.pop()
        last_time = t

    total = max(e["time"] for e in events) - min(e["time"] for e in events)
    if total == 0:
        return {}
    return {k: (v / total) * 100.0 for k, v in exclusive_time.items()}

def analyze_torch(events, flops_per_thread):
    tflops_sums = {}
    counts = {}
    for ev in events:
        if ev.get("cat") == "kernel":
            name = ev["name"]
            args = ev.get("args", {})
            dur_us = ev.get("dur", 0.0)
            if name in flops_per_thread and dur_us > 0:
                grid = args.get("Grid X", 1) * args.get("Grid Y", 1) * args.get("Grid Z", 1)
                block = args.get("Block X", 1) * args.get("Block Y", 1) * args.get("Block Z", 1)
                threads = grid * block
                flops = threads * flops_per_thread[name]
                tflops = flops / (dur_us * 1e-6) / 1e12
                tflops_sums[name] = tflops_sums.get(name, 0.0) + tflops
                counts[name] = counts.get(name, 0) + 1

    return {k: tflops_sums[k] / counts[k] for k in counts}

NCU_FIXTURE = [
    {"name": "matmul_kernel", "sm_pct": 92.5, "mem_pct": 45.0, "warps_pct": 80.0},
    {"name": "rmsnorm_kernel", "sm_pct": 20.0, "mem_pct": 85.5, "warps_pct": 90.0},
    {"name": "rotary_emb", "sm_pct": 10.0, "mem_pct": 20.0, "warps_pct": 45.0},
    {"name": "fused_add", "sm_pct": 40.0, "mem_pct": 40.0, "warps_pct": 75.0}
]

PROTON_FIXTURE = [
    {"time": 0.0, "type": "enter", "region": "step"},
    {"time": 10.0, "type": "enter", "region": "fwd"},
    {"time": 50.0, "type": "exit", "region": "fwd"},
    {"time": 60.0, "type": "enter", "region": "bwd"},
    {"time": 100.0, "type": "exit", "region": "bwd"},
    {"time": 110.0, "type": "exit", "region": "step"}
]

TORCH_FIXTURE = [
    {"name": "gemm_kernel", "cat": "kernel", "dur": 1500.0, "args": {"Grid X": 1024, "Grid Y": 1, "Grid Z": 1, "Block X": 128, "Block Y": 1, "Block Z": 1}},
    {"name": "gemm_kernel", "cat": "kernel", "dur": 1600.0, "args": {"Grid X": 1024, "Grid Y": 1, "Grid Z": 1, "Block X": 128, "Block Y": 1, "Block Z": 1}},
    {"name": "norm_kernel", "cat": "kernel", "dur": 50.0, "args": {"Grid X": 256, "Block X": 256}},
    {"name": "cpu_overhead", "cat": "cpu_op", "dur": 10.0}
]

TORCH_FLOPS = {
    "gemm_kernel": 16384.0,
    "norm_kernel": 256.0
}
