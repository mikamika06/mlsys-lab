import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness.ref as ref


def check(workdir):
    from triton_metrics.parser import parse_prometheus_text
    from triton_metrics.aggregator import compute_model_request_summary, compute_gpu_utilization_summary

    text = ref.generate_prometheus_payload(seed=202)
    samples = parse_prometheus_text(text)

    want_model = ref.compute_model_request_summary(samples)
    got_model = compute_model_request_summary(samples)

    want_gpu = ref.compute_gpu_utilization_summary(samples)
    got_gpu = compute_gpu_utilization_summary(samples)

    out = {"model_stats_match": 0.0, "gpu_stats_match": 0.0}

    model_ok = True
    if set(want_model.keys()) != set(got_model.keys()):
        model_ok = False
        out["_note"] = f"Model set mismatch: want {set(want_model.keys())}, got {set(got_model.keys())}"
    else:
        for m, want_val in want_model.items():
            got_val = got_model[m]
            if abs(want_val["success_count"] - got_val["success_count"]) > 1e-4 or \
               abs(want_val["avg_compute_time_ms"] - got_val["avg_compute_time_ms"]) > 1e-4:
                model_ok = False
                out["_note"] = f"Model {m} mismatch: want {want_val}, got {got_val}"
                break

    if model_ok:
        out["model_stats_match"] = 1.0

    gpu_ok = True
    if set(want_gpu.keys()) != set(got_gpu.keys()):
        gpu_ok = False
        if "_note" not in out:
            out["_note"] = f"GPU set mismatch: want {set(want_gpu.keys())}, got {set(got_gpu.keys())}"
    else:
        for g, want_val in want_gpu.items():
            got_val = got_gpu[g]
            if abs(want_val - got_val) > 1e-4:
                gpu_ok = False
                if "_note" not in out:
                    out["_note"] = f"GPU {g} mismatch: want {want_val}, got {got_val}"
                break

    if gpu_ok:
        out["gpu_stats_match"] = 1.0

    return out
