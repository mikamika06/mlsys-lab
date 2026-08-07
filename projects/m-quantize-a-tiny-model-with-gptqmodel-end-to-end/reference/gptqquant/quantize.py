import numpy as np

def quantize_weights(weight, config):
    w = weight.astype(np.float32)
    out_features, in_features = w.shape
    bits = config.bits
    group_size = config.group_size if config.group_size > 0 else in_features
    max_val = (1 << (bits - 1)) - 1 if config.sym else (1 << bits) - 1
    min_val = -(1 << (bits - 1)) if config.sym else 0
    scales = []
    zeros = []
    quantized_chunks = []
    for i in range(0, in_features, group_size):
        chunk = w[:, i:i+group_size]
        if config.sym:
            mx = np.max(np.abs(chunk), axis=1, keepdims=True)
            scale = np.maximum(mx / max_val, 1e-8)
            zero = np.zeros_like(scale)
            q = np.round(chunk / scale)
            q = np.clip(q, min_val, max_val)
        else:
            mn = np.min(chunk, axis=1, keepdims=True)
            mx = np.max(chunk, axis=1, keepdims=True)
            scale = np.maximum((mx - mn) / max_val, 1e-8)
            zero = np.round(-mn / scale)
            q = np.round(chunk / scale) + zero
            q = np.clip(q, 0, max_val)
        scales.append(scale)
        zeros.append(zero)
        quantized_chunks.append(q.astype(np.int32))
    quantized_weight = np.hstack(quantized_chunks)
    scales = np.hstack(scales)
    zeros = np.hstack(zeros)
    return quantized_weight, scales, zeros
