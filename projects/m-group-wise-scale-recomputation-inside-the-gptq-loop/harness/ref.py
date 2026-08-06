import numpy as np

np.random.seed(42)
W_TEST = np.random.randn(32, 64).astype(np.float32)
H_TEST = np.dot(W_TEST.T, W_TEST) + np.eye(64, dtype=np.float32) * 0.1
GROUP_SIZE = 16
BITS = 4


def compute_group_scales(w: np.ndarray, group_size: int, bits: int) -> np.ndarray:
    max_val = float(2 ** (bits - 1) - 1)
    rows, cols = w.shape
    num_groups = (cols + group_size - 1) // group_size
    scales = np.zeros((rows, num_groups), dtype=np.float32)
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, cols)
        sub = w[:, start:end]
        maxs = np.max(np.abs(sub), axis=1)
        scales[:, g] = np.maximum(maxs / max_val, 1e-8)
    return scales


def gptq_quantize_with_recompute(w: np.ndarray, h: np.ndarray, group_size: int, bits: int) -> np.ndarray:
    w_q = w.copy()
    rows, cols = w.shape
    max_val = float(2 ** (bits - 1) - 1)
    invh = np.linalg.inv(h)
    for i in range(cols):
        current_scales = compute_group_scales(w_q, group_size, bits)
        col_w = w_q[:, i]
        g_idx = i // group_size
        scale = current_scales[:, g_idx]
        q = np.round(col_w / scale)
        q = np.clip(q, -max_val, max_val)
        q_w = q * scale
        err = (col_w - q_w) / invh[i, i]
        w_q[:, i] = q_w
        if i + 1 < cols:
            w_q[:, i+1:] -= np.outer(err, invh[i, i+1:])
    return w_q
