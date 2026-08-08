import numpy as np


def get_test_tensors():
    np.random.seed(42)
    t1 = np.linspace(-1.5, 1.5, 64, dtype=np.float32)
    t2 = (np.random.randn(2, 64) * 0.5).astype(np.float32)
    t3 = np.zeros(128, dtype=np.float32)
    return [t1, t2, t3]


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


def get_reference_quantizations():
    return [quantize(t) for t in get_test_tensors()]
