import numpy as np
from quantizer.qparams import derive_qparams


def quantize_and_dequantize(tensor, granularity, symmetric=True, num_bits=8, group_size=None):
    arr = np.asarray(tensor, dtype=np.float32)
    scale, zero_point = derive_qparams(arr, granularity, symmetric=symmetric, num_bits=num_bits, group_size=group_size)

    qmin = -(1 << (num_bits - 1)) if symmetric else 0
    qmax = (1 << (num_bits - 1)) - 1 if symmetric else (1 << num_bits) - 1

    if granularity == "per_tensor":
        s = scale
        zp = zero_point
    elif granularity == "per_channel":
        s = scale[:, None]
        zp = zero_point[:, None]
    elif granularity in ("block_wise", "sub_vector"):
        s = scale.reshape(-1, 1)
        zp = zero_point.reshape(-1, 1)
        arr_flat = arr.reshape(-1, group_size)
        q = np.clip(np.round(arr_flat / s) + zp, qmin, qmax)
        deq = (q - zp) * s
        return deq.reshape(arr.shape)

    q = np.clip(np.round(arr / s) + zp, qmin, qmax)
    deq = (q - zp) * s
    return deq
