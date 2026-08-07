import ref
import numpy as np


def check(workdir):
    from quanteval.eval import compute_ir_size_reduction, compute_benchmark_latency_gain
    out = {"size_ratio_match": 0.0, "latency_gain_match": 0.0}
    size_ok = 0
    latency_ok = 0
    for case in ref.CASES:
        want_size = ref.compute_size_reduction(case["fp32_size"], case["int8_size"])
        got_size = compute_ir_size_reduction(case["fp32_size"], case["int8_size"])
        if np.isclose(got_size, want_size, rtol=1e-5, atol=1e-5):
            size_ok += 1

        want_lat = ref.compute_latency_gain(case["fp32_latency"], case["int8_latency"])
        got_lat = compute_benchmark_latency_gain(case["fp32_latency"], case["int8_latency"])
        if np.isclose(got_lat, want_lat, rtol=1e-5, atol=1e-5):
            latency_ok += 1

    if size_ok == len(ref.CASES):
        out["size_ratio_match"] = 1.0
    if latency_ok == len(ref.CASES):
        out["latency_gain_match"] = 1.0
    return out
