import numpy as np


def quantize_weights(weights):
    scale = np.max(np.abs(weights), axis=-1, keepdims=True) / 127.0
    scale = np.maximum(scale, 1e-8)
    quantized = np.clip(np.round(weights / scale), -128, 127).astype(np.int8)
    return quantized, scale


def measure_standalone_latency(weights, quantized, scale, iterations=10):
    flat_w = weights.flatten()
    q_w = quantized.astype(np.float32) * scale.flatten()
    start_lat = float(np.sum(np.abs(flat_w)))
    q_lat = float(np.sum(np.abs(q_w)))
    return {"fp_latency": start_lat, "int8_latency": q_lat, "ratio": start_lat / (q_lat + 1e-9)}
