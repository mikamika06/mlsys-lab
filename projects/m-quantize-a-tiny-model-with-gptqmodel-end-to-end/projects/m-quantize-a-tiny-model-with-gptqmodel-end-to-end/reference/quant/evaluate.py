import numpy as np


def compute_size_ratio(model, artifact):
    orig = artifact["original_bytes"]
    packed = artifact["packed_bytes"]
    return float(orig / packed)


def evaluate_error(model, artifact):
    w = model["weight"]
    q = artifact["quantized_weight"]
    scale = artifact["scale"]
    w_recons = q.astype(np.float32) * scale
    mse = float(np.mean((w - w_recons) ** 2))
    return mse < 0.1
