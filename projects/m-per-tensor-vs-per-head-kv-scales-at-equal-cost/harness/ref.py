import numpy as np


def get_test_tensors():
    np.random.seed(42)
    return [np.random.randn(8, 128).astype(np.float32) for _ in range(3)]


def compute_ref_scales(tensor, mode):
    if mode == "per-tensor":
        amax = np.max(np.abs(tensor))
        scale = amax / 7.0 if amax > 0 else 1.0
        return np.array([scale], dtype=np.float32)
    else:
        amax = np.max(np.abs(tensor), axis=-1, keepdims=True)
        scales = amax / 7.0
        scales[scales == 0] = 1.0
        return scales.astype(np.float32)


def compute_ref_allocation(w, k, b):
    best_w, best_kv, min_diff = 8, 4, float("inf")
    for wb in [2, 4, 8]:
        for kb in [2, 4, 8]:
            cost = w * wb + k * kb
            diff = abs(cost - b)
            if diff < min_diff:
                min_diff = diff
                best_w, best_kv = wb, kb
    return {"weight_bits": best_w, "kv_bits": best_kv}


def compute_ref_decision(s, e, b):
    est = b * (1.0 + 0.0001 * s)
    return bool(est <= e)
