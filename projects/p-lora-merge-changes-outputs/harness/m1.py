import numpy as np
from harness import ref

def check(workdir):
    from lora_merge.analyzer import measure_layer_diff
    bw, la, lb, alpha, rank, x = ref.generate_test_data()
    res = measure_layer_diff(bw, la, lb, alpha, rank, x)
    m = {"layers_analyzed": 0.0, "diff_detected": 0.0}
    if isinstance(res, dict) and "layers_analyzed" in res:
        m["layers_analyzed"] = float(res["layers_analyzed"])
        m["diff_detected"] = float(res["diff_detected"])
    return m
