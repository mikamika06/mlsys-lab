import numpy as np

RNG = np.random.RandomState(42)

TEST_DATASETS = [
    {
        "tensor": RNG.randn(128).astype(np.float32) * 5.0,
        "group_size": 32,
        "asymmetric": False
    },
    {
        "tensor": RNG.uniform(-10.0, 20.0, size=(16, 16)).astype(np.float32),
        "group_size": 16,
        "asymmetric": True
    },
    {
        "tensor": np.array([-15.0, -2.0, 0.5, 3.1, 8.0, 14.2], dtype=np.float32),
        "group_size": 4,
        "asymmetric": False
    }
]


def quantize_group_int4(tensor, group_size, asymmetric=False):
    shape = tensor.shape
    flat = tensor.astype(np.float32).reshape(-1)
    num_elements = flat.size
    pad_len = (group_size - (num_elements % group_size)) % group_size
    if pad_len > 0:
        flat_padded = np.pad(flat, (0, pad_len), mode='constant', constant_values=0.0)
    else:
        flat_padded = flat

    num_groups = flat_padded.size // group_size
    grouped = flat_padded.reshape(num_groups, group_size)

    if asymmetric:
        g_min = np.min(grouped, axis=1, keepdims=True)
        g_max = np.max(grouped, axis=1, keepdims=True)
        scale = (g_max - g_min) / 15.0
        scale = np.where(scale == 0.0, 1.0, scale)
        zero_point = np.round(-g_min / scale) - 8.0
        zero_point = np.clip(zero_point, -8, 7)
        q = np.round(grouped / scale) + zero_point
        q = np.clip(q, -8, 7).astype(np.int8)
    else:
        abs_max = np.max(np.abs(grouped), axis=1, keepdims=True)
        scale = abs_max / 7.0
        scale = np.where(scale == 0.0, 1.0, scale)
        zero_point = np.zeros_like(scale, dtype=np.float32)
        q = np.round(grouped / scale)
        q = np.clip(q, -8, 7).astype(np.int8)

    q_flat = q.reshape(-1)[:num_elements].reshape(shape)
    scale_flat = scale.reshape(-1)
    zp_flat = zero_point.reshape(-1)

    return q_flat, scale_flat, zp_flat


def dequantize_group_int4(qtensor, scale, zero_point, group_size):
    shape = qtensor.shape
    q_flat = qtensor.reshape(-1).astype(np.float32)
    num_elements = q_flat.size
    pad_len = (group_size - (num_elements % group_size)) % group_size
    if pad_len > 0:
        q_padded = np.pad(q_flat, (0, pad_len), mode='constant', constant_values=0.0)
    else:
        q_padded = q_flat

    num_groups = q_padded.size // group_size
    grouped_q = q_padded.reshape(num_groups, group_size)
    scale_col = scale.reshape(num_groups, 1)
    zp_col = zero_point.reshape(num_groups, 1)

    recon_grouped = (grouped_q - zp_col) * scale_col
    recon_flat = recon_grouped.reshape(-1)[:num_elements]
    return recon_flat.reshape(shape)


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
