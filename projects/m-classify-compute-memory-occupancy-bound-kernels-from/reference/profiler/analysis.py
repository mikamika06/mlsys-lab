import json


def classify_ncu_kernel(ncu_text: str) -> str:
    compute_throughput = 0.0
    memory_throughput = 0.0
    achieved_occupancy = 0.0
    
    for line in ncu_text.splitlines():
        if "sm__throughput.avg.pct_of_peak_sustained_elapsed" in line:
            parts = line.split()
            try:
                compute_throughput = float(parts[-1])
            except ValueError:
                pass
        elif "dram__throughput.avg.pct_of_peak_sustained_elapsed" in line:
            parts = line.split()
            try:
                memory_throughput = float(parts[-1])
            except ValueError:
                pass
        elif "sm__warps_active.avg.pct_of_peak_sustained_active" in line:
            parts = line.split()
            try:
                achieved_occupancy = float(parts[-1])
            except ValueError:
                pass

    if achieved_occupancy < 40.0 and achieved_occupancy <= compute_throughput and achieved_occupancy <= memory_throughput:
        return "occupancy-bound"
    elif compute_throughput > memory_throughput:
        return "compute-bound"
    else:
        return "memory-bound"


def compute_proton_breakdown(proton_lines: list) -> dict:
    region_times = {}
    total_time = 0.0
    for line in proton_lines:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            region = parts[0].strip()
            try:
                t = float(parts[1].strip())
            except ValueError:
                continue
            region_times[region] = region_times.get(region, 0.0) + t
            total_time += t

    if total_time == 0.0:
        return {}
    
    result = {}
    for region, t in region_times.items():
        result[region] = round((t / total_time) * 100.0, 2)
    return result


def parse_torch_trace_kernel(trace_json: dict) -> dict:
    args = trace_json.get("args", {})
    block_size = [
        args.get("block X", args.get("blockSizeX", 1)),
        args.get("block Y", args.get("blockSizeY", 1)),
        args.get("block Z", args.get("blockSizeZ", 1)),
    ]
    grid_size = [
        args.get("grid X", args.get("gridSizeX", 1)),
        args.get("grid Y", args.get("gridSizeY", 1)),
        args.get("grid Z", args.get("gridSizeZ", 1)),
    ]
    dur_us = trace_json.get("dur", 0.0)
    flops = args.get("flops", 0.0)
    
    tflops = 0.0
    if dur_us > 0:
        tflops = (flops / (dur_us * 1e-6)) / 1e12

    return {
        "block_size": block_size,
        "grid_size": grid_size,
        "tflops": round(tflops, 4),
    }
