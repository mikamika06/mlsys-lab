import numpy as np
from quant.group_quant import quantize_group_int4, dequantize_group_int4


def compute_reconstruction_mse(original, reconstructed):
    diff = original.astype(np.float32) - reconstructed.astype(np.float32)
    return float(np.mean(diff ** 2))


def classify_saturation(tensor, group_size, asymmetric=False):
    shape = tensor.shape
    qtensor, scale, zero_point = quantize_group_int4(tensor, group_size, asymmetric=asymmetric)
    q_flat = qtensor.reshape(-1)

    saturated_mask = (q_flat == -8) | (q_flat == 7)
    saturated_count = int(np.sum(saturated_mask))
    unsaturated_count = int(q_flat.size - saturated_count)

    reconstructed = dequantize_group_int4(qtensor, scale, zero_point, group_size)
    mse = compute_reconstruction_mse(tensor, reconstructed)

    return {
        "mse": mse,
        "saturated_count": saturated_count,
        "unsaturated_count": unsaturated_count,
        "total_count": int(q_flat.size),
        "saturation_ratio": float(saturated_count / q_flat.size)
    }
