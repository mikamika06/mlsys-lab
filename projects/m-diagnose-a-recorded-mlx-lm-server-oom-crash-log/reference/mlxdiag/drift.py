import numpy as np


def simulate_quant_dequant(weights, bits=4, group_size=64):
    """Simulates round-trip affine symmetric quantization and dequantization."""
    weights = np.array(weights, dtype=np.float32)
    orig_shape = weights.shape
    flat = weights.flatten()

    pad_len = (group_size - (flat.size % group_size)) % group_size
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), mode="constant")

    reshaped = flat.reshape(-1, group_size)
    max_val = np.max(np.abs(reshaped), axis=1, keepdims=True)
    max_val = np.maximum(max_val, 1e-8)

    qmax = (1 << (bits - 1)) - 1
    scales = max_val / qmax

    q = np.round(reshaped / scales)
    q = np.clip(q, -qmax, qmax)

    dq = q * scales
    dq_flat = dq.reshape(-1)[: weights.size]
    return dq_flat.reshape(orig_shape)


def evaluate_weight_drift(weights, bits=4, group_size=64, max_allowed_mse=0.05):
    """Calculates MSE weight drift between original and round-trip weights."""
    weights = np.array(weights, dtype=np.float32)
    dequant = simulate_quant_dequant(weights, bits=bits, group_size=group_size)
    mse = float(np.mean((weights - dequant) ** 2))
    return {
        "mse": mse,
        "exceeds_threshold": mse > max_allowed_mse,
        "max_allowed_mse": float(max_allowed_mse),
    }
