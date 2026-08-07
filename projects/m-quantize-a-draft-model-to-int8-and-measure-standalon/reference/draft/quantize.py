import numpy as np


def quantize_weights_int8(weights):
    w = np.asarray(weights, dtype=np.float32)
    max_val = np.max(np.abs(w), axis=-1, keepdims=True)
    scale = np.maximum(max_val, 1e-8) / 127.0
    qweights = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return {"qweights": qweights, "scale": scale.astype(np.float32)}


def measure_standalone_latency(fp16_weights, int8_data, inputs, iterations=20):
    w_fp16 = np.asarray(fp16_weights, dtype=np.float32)
    x = np.asarray(inputs, dtype=np.float32)
    qweights = int8_data["qweights"]
    scale = int8_data["scale"]

    fp16_ops = 0
    for _ in range(iterations):
        _ = np.dot(x, w_fp16.T)
        fp16_ops += x.shape[0] * w_fp16.shape[0] * w_fp16.shape[1] * 2

    int8_ops = 0
    for _ in range(iterations):
        dequant = qweights.astype(np.float32) * scale
        _ = np.dot(x, dequant.T)
        int8_ops += x.shape[0] * qweights.shape[0] * qweights.shape[1]

    int8_bytes = (x.nbytes + qweights.nbytes + scale.nbytes) * iterations
    fp16_bytes = (x.nbytes + w_fp16.nbytes) * iterations
    latency_ratio = int8_bytes / float(fp16_bytes)

    dequant_w = qweights.astype(np.float32) * scale
    max_err = float(np.max(np.abs(w_fp16 - dequant_w)))

    return {
        "latency_ratio": float(latency_ratio),
        "max_error": max_err,
        "fp16_ops": fp16_ops,
        "int8_ops": int8_ops
    }
