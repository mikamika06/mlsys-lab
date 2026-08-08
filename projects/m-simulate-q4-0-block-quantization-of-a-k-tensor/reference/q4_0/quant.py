import numpy as np


def quantize(tensor):
    arr = np.asarray(tensor, dtype=np.float32)
    shape = arr.shape
    flat = arr.reshape(-1)
    if flat.size % 32 != 0:
        raise ValueError("Tensor size must be a multiple of 32")
    blocks = flat.reshape(-1, 32)

    max_val = np.max(np.abs(blocks), axis=1)
    d = max_val / -8.0
    d = np.where(d == 0, np.float32(1e-7), d.astype(np.float32))

    scaled = blocks / d[:, None]
    v = np.round(scaled) + 8
    ids = np.clip(v, 0, 15).astype(np.uint8)

    low = ids[:, 0::2]
    high = ids[:, 1::2]
    packed = np.bitwise_or(low, np.left_shift(high, 4))

    return {"shape": shape, "scales": d, "packed": packed}


def dequantize(q_dict):
    shape = q_dict["shape"]
    d = q_dict["scales"]
    packed = q_dict["packed"]

    low = np.bitwise_and(packed, 0x0F)
    high = np.bitwise_and(np.right_shift(packed, 4), 0x0F)

    ids = np.empty((packed.shape[0], 32), dtype=np.float32)
    ids[:, 0::2] = low
    ids[:, 1::2] = high

    blocks = (ids - 8) * d[:, None]
    return blocks.reshape(shape)
