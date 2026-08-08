import numpy as np


def quantize_int4(x, group_size, asymmetric=True):
    shape = x.shape
    x_flat = x.reshape(-1)
    num_elements = x_flat.size
    num_groups = num_elements // group_size
    x_grouped = x_flat.reshape(num_groups, group_size)

    qmin = 0 if asymmetric else -8
    qmax = 15 if asymmetric else 7

    scales = np.zeros(num_groups, dtype=np.float32)
    zero_points = np.zeros(num_groups, dtype=np.int32)
    q_grouped = np.zeros((num_groups, group_size), dtype=np.int32)

    for g in range(num_groups):
        g_data = x_grouped[g]
        min_val = float(np.min(g_data))
        max_val = float(np.max(g_data))

        if asymmetric:
            if max_val == min_val:
                scale = 1.0
                zp = qmin
            else:
                scale = (max_val - min_val) / (qmax - qmin)
                zp = int(np.round(qmin - min_val / scale))
                zp = int(np.clip(zp, qmin, qmax))
            q = np.round(g_data / scale) + zp
        else:
            max_abs = max(abs(min_val), abs(max_val))
            if max_abs == 0:
                scale = 1.0
            else:
                scale = max_abs / qmax
            zp = 0
            q = np.round(g_data / scale)

        q = np.clip(q, qmin, qmax).astype(np.int32)
        scales[g] = scale
        zero_points[g] = zp
        q_grouped[g] = q

    return q_grouped.reshape(shape), scales, zero_points


def dequantize_int4(q_tensor, scales, zero_points, group_size, asymmetric=True):
    shape = q_tensor.shape
    q_flat = q_tensor.reshape(-1)
    num_elements = q_flat.size
    num_groups = num_elements // group_size
    q_grouped = q_flat.reshape(num_groups, group_size)

    recon_grouped = np.zeros((num_groups, group_size), dtype=np.float32)
    for g in range(num_groups):
        scale = scales[g]
        zp = zero_points[g]
        if asymmetric:
            recon_grouped[g] = (q_grouped[g].astype(np.float32) - zp) * scale
        else:
            recon_grouped[g] = q_grouped[g].astype(np.float32) * scale

    return recon_grouped.reshape(shape)
