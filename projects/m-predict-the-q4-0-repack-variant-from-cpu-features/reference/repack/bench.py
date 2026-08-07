import time
import numpy as np
from repack.predict import predict_variant

def benchmark_repack(weights, features):
    variant = predict_variant(features)
    t0 = time.perf_counter()
    if variant != "q4_0_scalar":
        repacked = np.ascontiguousarray(weights[::-1])
    else:
        repacked = np.copy(weights)
    t1 = time.perf_counter()
    duration = max(t1 - t0, 1e-6)

    t2 = time.perf_counter()
    _ = np.copy(weights)
    t3 = time.perf_counter()
    base_duration = max(t3 - t2, 1e-6)

    speedup = base_duration / duration
    return {
        "variant": variant,
        "duration": duration,
        "baseline_duration": base_duration,
        "speedup": speedup,
        "repacked": repacked
    }
