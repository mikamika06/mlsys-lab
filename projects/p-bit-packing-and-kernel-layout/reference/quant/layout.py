import numpy as np

def describe_layout():
    return {
        "block_size": 32,
        "bits": 4,
        "swizzle": True,
        "description": "Kernel expects weights packed into 32-bit words with 4 bits per value, swizzled by thread warp layout."
    }

def transform_layout(tensor):
    arr = np.asarray(tensor, dtype=np.int32)
    flat = arr.flatten()
    if len(flat) % 8 != 0:
        pad_len = (8 - len(flat) % 8) % 8
        flat = np.pad(flat, (0, pad_len), mode='constant')
    reshaped = flat.reshape(-1, 8)
    swizzled = reshaped[:, [0, 4, 1, 5, 2, 6, 3, 7]]
    return swizzled.flatten()
