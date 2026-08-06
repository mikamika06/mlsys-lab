import random

RUNS = [
    {"run_id": "r1", "engine": "openvino", "threads_allocated": 4, "physical_cores": 8, "contention_score": 0.1, "tokens_per_sec": 45.2},
    {"run_id": "r2", "engine": "onnxruntime", "threads_allocated": 12, "physical_cores": 8, "contention_score": 0.85, "tokens_per_sec": 18.5},
    {"run_id": "r3", "engine": "openvino", "threads_allocated": 8, "physical_cores": 8, "contention_score": 0.2, "tokens_per_sec": 52.0},
    {"run_id": "r4", "engine": "onnxruntime", "threads_allocated": 4, "physical_cores": 8, "contention_score": 0.15, "tokens_per_sec": 42.1},
    {"run_id": "r5", "engine": "openvino", "threads_allocated": 16, "physical_cores": 8, "contention_score": 0.92, "tokens_per_sec": 22.0},
]


def classify_run(run):
    if run["threads_allocated"] <= run["physical_cores"] and run["contention_score"] < 0.5:
        return "fair"
    return "unfair"


def classify_runs(runs):
    return [classify_run(r) for r in runs]


def compute_metrics(runs):
    ov = [r["tokens_per_sec"] for r in runs if r["engine"] == "openvino"]
    ort = [r["tokens_per_sec"] for r in runs if r["engine"] == "onnxruntime"]
    avg_ov = sum(ov) / len(ov) if ov else 0.0
    avg_ort = sum(ort) / len(ort) if ort else 0.0
    ratio = avg_ov / avg_ort if avg_ort > 0 else 0.0
    return {"avg_openvino": avg_ov, "avg_onnxruntime": avg_ort, "throughput_ratio": ratio}
