import os
import ref

def check(workdir):
    m = {"speedup_ratio": 0.0}
    path = ref.create_dummy_model(workdir)
    out_path = os.path.join(workdir, "optimized.tflite")
    try:
        from edge.partitioner import measure_latency_ratio
        ratio = measure_latency_ratio(path, out_path)
        m["speedup_ratio"] = float(ratio)
    except Exception:
        pass
    return m
