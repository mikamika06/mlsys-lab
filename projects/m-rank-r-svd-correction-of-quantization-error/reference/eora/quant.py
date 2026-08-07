import numpy as np


def quantize_weights(weights, method="rtn"):
    w = np.asarray(weights, dtype=np.float32)
    if method == "rtn":
        scale = np.max(np.abs(w)) / 7.0
        scale = max(scale, 1e-5)
        q = np.clip(np.round(w / scale), -8, 7)
        return q * scale
    elif method == "gptq":
        scale = np.max(np.abs(w)) / 7.0
        scale = max(scale, 1e-5)
        q = np.clip(np.round(w / scale), -8, 7)
        return q * scale
    elif method == "rotation_gptq":
        np.random.seed(42)
        h, d = w.shape
        q_mat, _ = np.linalg.qr(np.random.randn(d, d))
        w_rot = w @ q_mat
        scale = np.max(np.abs(w_rot)) / 7.0
        scale = max(scale, 1e-5)
        q = np.clip(np.round(w_rot / scale), -8, 7)
        return (q * scale) @ q_mat.T
    elif method == "learned_rounding":
        scale = np.max(np.abs(w)) / 7.0
        scale = max(scale, 1e-5)
        floor_w = np.floor(w / scale)
        frac = w / scale - floor_w
        q = floor_w + (frac > 0.4).astype(np.float32)
        q = np.clip(q, -8, 7)
        return q * scale
    else:
        raise ValueError(f"Unknown method {method}")
