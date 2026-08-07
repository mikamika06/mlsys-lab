import numpy as np


def create_tiny_model():
    np.random.seed(42)
    return {
        "weight": np.random.randn(64, 64).astype(np.float32),
        "bias": np.zeros((64,), dtype=np.float32),
    }


def get_calibration_data():
    np.random.seed(42)
    return [np.random.randn(16, 64).astype(np.float32) for _ in range(5)]


def build_calibration_dataset(model, inputs):
    activations = []
    w = model["weight"]
    for x in inputs:
        out = x @ w.T
        activations.append(out)
    return {"inputs": inputs, "activations": activations}


def quantize_weights(model, calib_data, bits=4):
    w = model["weight"]
    scale = np.max(np.abs(w), axis=1, keepdims=True) / 7.0
    scale = np.maximum(scale, 1e-5)
    q = np.round(w / scale).astype(np.int32)
    q = np.clip(q, -8, 7)
    packed_bytes = (w.size * bits) // 8
    original_bytes = w.nbytes
    return {
        "quantized_weight": q,
        "scale": scale,
        "bits": bits,
        "original_bytes": original_bytes,
        "packed_bytes": packed_bytes,
    }


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
