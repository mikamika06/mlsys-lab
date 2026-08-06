import json
import random


def generate_cases(seed=42):
    rng = random.Random(seed)
    cases = []
    for i in range(5):
        def_lat = rng.uniform(10.0, 50.0)
        auto_lat = def_lat * rng.uniform(0.6, 0.95)
        def_comp = rng.uniform(5.0, 15.0)
        auto_comp = def_comp * rng.uniform(2.0, 5.0)

        default_record = {"latency_ms": def_lat, "compile_time_s": def_comp}
        autotune_record = {"latency_ms": auto_lat, "compile_time_s": auto_comp}

        from autotune_metrics.analyzer import compare_latencies
        expected = compare_latencies(default_record, autotune_record)
        cases.append({
            "default_record": default_record,
            "autotune_record": autotune_record,
            "expected": expected
        })
    return cases


def generate_logs(seed=42):
    rng = random.Random(seed)
    lines = []
    best_c = float("inf")
    best_cfg = None
    for i in range(10):
        cost = rng.uniform(0.5, 5.0)
        cfg = f"block_{i}_threads_{rng.choice([128, 256, 512])}"
        if cost < best_c:
            best_c = cost
            best_cfg = cfg
        lines.append(json.dumps({"config": cfg, "cost_ms": cost}))
    return lines, best_cfg


def generate_traces(seed=42):
    events = [
        {"name": "setup", "ph": "X"},
        {"name": "cuda_graph_recapture", "ph": "R"},
        {"name": "compute", "ph": "X"}
    ]
    return events, [1]
