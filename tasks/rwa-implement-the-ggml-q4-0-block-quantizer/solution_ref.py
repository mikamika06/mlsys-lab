import numpy as np


def q4_0_quantize(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    blocks = x.reshape(-1, 32)
    scales = np.empty(blocks.shape[0], dtype=np.float16)
    codes = np.empty((blocks.shape[0], 16), dtype=np.uint8)

    for b, block in enumerate(blocks):
        d = np.max(np.abs(block)) / -8.0
        scales[b] = np.float16(d)
        q = np.round(block / d).astype(np.int32)
        q = np.clip(q, -8, 7)
        nib = (q + 8).astype(np.uint8)
        codes[b] = nib[::2] | (nib[1::2] << 4)

    return scales, codes
