import numpy as np


def derive_qparams(tensor, granularity, symmetric=True, num_bits=8, group_size=None):
    arr = np.asarray(tensor, dtype=np.float32)
    qmin = -(1 << (num_bits - 1)) if symmetric else 0
    qmax = (1 << (num_bits - 1)) - 1 if symmetric else (1 << num_bits) - 1

    if granularity == "per_tensor":
        reshaped = arr.reshape(1, -1)
        scale_shape = (1,)
    elif granularity == "per_channel":
        reshaped = arr.reshape(arr.shape[0], -1)
        scale_shape = (arr.shape[0], 1)
    elif granularity == "block_wise":
        if group_size is None:
            raise ValueError("group_size required for block_wise")
        reshaped = arr.reshape(-1, group_size)
        scale_shape = (-1, 1)
    elif granularity == "sub_vector":
        if group_size is None:
            raise ValueError("group_size required for sub_vector")
        reshaped = arr.reshape(-1, group_size)
        scale_shape = (arr.shape[0], -1)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    if symmetric:
        max_val = np.abs(reshaped).max(axis=-1)
        scale = max_val / float(qmax)
        scale = np.where(scale == 0, 1.0, scale)
        zero_point = np.zeros_like(scale, dtype=np.int32)
    else:
        min_val = reshaped.min(axis=-1)
        max_val = reshaped.max(axis=-1)
        scale = (max_val - min_val) / float(qmax - qmin)
        scale = np.where(scale == 0, 1.0, scale)
        zero_point = np.round(qmin - min_val / scale).astype(np.int32)
        zero_point = np.clip(zero_point, qmin, qmax)

    if granularity == "per_tensor":
        scale = scale.reshape(1)
        zero_point = zero_point.reshape(1)
    elif granularity == "per_channel":
        scale = scale.reshape(arr.shape[0])
        zero_point = zero_point.reshape(arr.shape[0])
    elif granularity == "block_wise":
        scale = scale.reshape(-1)
        zero_point = zero_point.reshape(-1)
    elif granularity == "sub_vector":
        num_blocks = arr.shape[1] // group_size
        scale = scale.reshape(arr.shape[0], num_blocks)
        zero_point = zero_point.reshape(arr.shape[0], num_blocks)

    return scale, zero_point
