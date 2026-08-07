import numpy as np

def measure_running_sum_error(data):
    d64 = np.asarray(data, dtype=np.float64)
    ref = np.cumsum(d64)

    f32 = np.cumsum(d64.astype(np.float32)).astype(np.float64)
    f16 = np.cumsum(d64.astype(np.float16)).astype(np.float64)

    err32 = np.abs(f32 - ref)
    err16 = np.abs(f16 - ref)

    max32 = float(np.max(err32))
    max16 = float(np.max(err16))

    ratio = float(err16[-1] / (err32[-1] + 1e-12))

    return {
        "max_err_fp32": max32,
        "max_err_fp16": max16,
        "final_err_fp32": float(err32[-1]),
        "final_err_fp16": float(err16[-1]),
        "drift_ratio": ratio,
    }
