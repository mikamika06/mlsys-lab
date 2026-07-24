import numpy as np

def compare_quantization(W):
    # Per‑tensor quantisation
    min_t = W.min()
    max_t = W.max()
    range_t = max_t - min_t
    if range_t == 0:
        dq_t = np.full_like(W, min_t, dtype=np.float64)
    else:
        scale_t = range_t / 255.0
        q_t = np.clip(np.round((W - min_t) / scale_t), 0, 255).astype(np.uint8)
        dq_t = q_t.astype(np.float64) * scale_t + min_t

    # Per‑channel quantisation
    min_c = W.min(axis=0)
    max_c = W.max(axis=0)
    range_c = max_c - min_c
    dq_c = np.empty_like(W, dtype=np.float64)

    zero_mask = range_c == 0
    if np.any(zero_mask):
        dq_c[:, zero_mask] = min_c[zero_mask]

    non_zero_mask = ~zero_mask
    if np.any(non_zero_mask):
        scale_c = range_c[non_zero_mask] / 255.0
        q_c = np.clip(np.round((W[:, non_zero_mask] - min_c[non_zero_mask]) / scale_c), 0, 255).astype(np.uint8)
        dq_c[:, non_zero_mask] = q_c.astype(np.float64) * scale_c + min_c[non_zero_mask]

    return dq_t, dq_c
