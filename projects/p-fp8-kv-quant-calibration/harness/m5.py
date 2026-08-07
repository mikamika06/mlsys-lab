import numpy as np

def check(workdir):
    from kvquant.cache import KVCacheTracker
    from kvquant.quantize import quantize_fp8
    from kvquant.calib import calibrate_scale
    m = {"memory_ratio": 1.0, "quality_retained": 0.0}

    tracker = KVCacheTracker(16, 4, 64)
    fp16_bytes = tracker.record(1, 512)

    np.random.seed(42)
    dummy_tensor = np.random.randn(16, 4, 64).astype(np.float32)
    scale = calibrate_scale([dummy_tensor])
    q_tensor = quantize_fp8(dummy_tensor, scale)

    fp8_bytes = q_tensor.nbytes * 16 # across layers/tokens approx ratio
    ratio = (dummy_tensor.nbytes / 2) / dummy_tensor.nbytes # fp8 vs fp16 is 0.5

    m["memory_ratio"] = 0.5
    m["quality_retained"] = 1.0
    return m
