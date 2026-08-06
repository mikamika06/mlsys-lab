import numpy as np
from quant.decomposed import decomposed_matmul


def find_optimal_threshold(x, w, max_fp16_flops):
    candidates = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    best_thresh = candidates[0]
    best_error = float("inf")
    ref_out = np.matmul(x.astype(np.float32), w.astype(np.float32))

    for t in candidates:
        outlier_count = np.sum(np.abs(x) > t)
        flops = outlier_count * w.shape[1]
        if flops <= max_fp16_flops:
            out = decomposed_matmul(x, w, t)
            err = np.mean((out - ref_out) ** 2)
            if err < best_error:
                best_error = err
                best_thresh = t
    return best_thresh
