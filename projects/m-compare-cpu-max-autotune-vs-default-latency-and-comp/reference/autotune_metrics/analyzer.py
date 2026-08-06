import json


def compare_latencies(default_record, autotune_record):
    def_lat = float(default_record["latency_ms"])
    auto_lat = float(autotune_record["latency_ms"])
    def_comp = float(default_record["compile_time_s"])
    auto_comp = float(autotune_record["compile_time_s"])

    latency_ratio = auto_lat / def_lat if def_lat > 0 else 0.0
    compile_time_ratio = auto_comp / def_comp if def_comp > 0 else 0.0

    return {
        "latency_ratio": round(latency_ratio, 4),
        "compile_time_ratio": round(compile_time_ratio, 4),
        "is_faster": auto_lat < def_lat
    }


def find_argmin_config(log_lines):
    best_cost = float("inf")
    best_config = None
    for line in log_lines:
        try:
            data = json.loads(line)
        except Exception:
            continue
        if "cost_ms" in data and "config" in data:
            cost = float(data["cost_ms"])
            if cost < best_cost:
                best_cost = cost
                best_config = data["config"]
    return best_config


def locate_cuda_graph_recapture(trace_events):
    recapture_indices = []
    for i, ev in enumerate(trace_events):
        name = str(ev.get("name", ""))
        ph = str(ev.get("ph", ""))
        if "cuda_graph_recapture" in name.lower() or "graph_capture" in name.lower() or ph == "R":
            recapture_indices.append(i)
    return recapture_indices
