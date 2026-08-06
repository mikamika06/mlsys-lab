import ref
import numpy as np


def check(workdir):
    from edgeonnx.export import simulate_export
    from edgeonnx.runner import measure_latency

    valid = 0.0
    for cfg in ref.CONFIGS:
        spec = simulate_export(cfg)
        inputs = np.array([1.0, 2.0], dtype=np.float32)
        lat_c = measure_latency(spec, inputs, "CoreMLExecutionProvider")
        lat_cpu = measure_latency(spec, inputs, "CPUExecutionProvider")
        if lat_c > 0 and lat_cpu > 0:
            valid = 1.0

    return {"latency_ratio_valid": valid}
