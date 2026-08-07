import numpy as np


def measure_reduction_error(steps):
    np.random.seed(42)
    vals = np.random.uniform(-0.01, 0.01, size=steps).astype(np.float64)
    ref_sum = np.sum(vals, dtype=np.float64)

    curr_fp32 = 0.0
    curr_fp16 = 0.0

    errors_fp32 = []
    errors_fp16 = []

    running_fp32 = 0.0
    running_fp16 = 0.0

    for i, v in enumerate(vals):
        running_fp32 = np.float32(running_fp32 + np.float32(v))
        running_fp16 = np.float16(running_fp16 + np.float16(v))

        if (i + 1) % max(1, steps // 10) == 0:
            errors_fp32.append(float(np.abs(running_fp32 - np.sum(vals[:i+1]))))
            errors_fp16.append(float(np.abs(running_fp16 - np.sum(vals[:i+1]))))

    return {
        "fp32_errors": errors_fp32,
        "fp16_errors": errors_fp16,
        "final_ref": float(ref_sum)
    }
