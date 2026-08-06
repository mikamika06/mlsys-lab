from typing import Dict, List
from triton_metrics.parser import MetricSample


def compute_model_request_summary(samples: List[MetricSample]) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, float] = {}
    total_time_us: Dict[str, float] = {}
    exec_counts: Dict[str, float] = {}

    for s in samples:
        model = s.labels.get("model")
        if not model:
            continue

        if s.name == "nv_inference_request_success":
            counts[model] = counts.get(model, 0.0) + s.value
        elif s.name == "nv_inference_compute_infer_time_us":
            total_time_us[model] = total_time_us.get(model, 0.0) + s.value
        elif s.name == "nv_inference_exec_count":
            exec_counts[model] = exec_counts.get(model, 0.0) + s.value

    models = set(counts.keys()) | set(total_time_us.keys())
    res = {}
    for m in sorted(models):
        c = counts.get(m, 0.0)
        t_us = total_time_us.get(m, 0.0)
        e = exec_counts.get(m, c if c > 0 else 1.0)
        avg_ms = (t_us / e / 1000.0) if e > 0 else 0.0
        res[m] = {
            "success_count": c,
            "avg_compute_time_ms": avg_ms
        }
    return res


def compute_gpu_utilization_summary(samples: List[MetricSample]) -> Dict[str, float]:
    gpu_totals: Dict[str, float] = {}
    gpu_counts: Dict[str, int] = {}

    for s in samples:
        if s.name == "nv_gpu_utilization":
            gpu_id = s.labels.get("gpu") or s.labels.get("gpu_uuid") or s.labels.get("device", "unknown")
            gpu_totals[gpu_id] = gpu_totals.get(gpu_id, 0.0) + s.value
            gpu_counts[gpu_id] = gpu_counts.get(gpu_id, 0) + 1

    return {gpu: gpu_totals[gpu] / gpu_counts[gpu] for gpu in sorted(gpu_totals.keys())}
