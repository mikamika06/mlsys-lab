import numpy as np


def generate_cases():
    np.random.seed(42)
    configs = []
    for _ in range(5):
        shape = (16, 32)
        arr = np.random.randn(*shape).astype(np.float32)
        flat = arr.reshape(-1, 4)
        for row in range(flat.shape[0]):
            idx = np.random.choice(4, size=2, replace=False)
            mask = np.zeros(4, dtype=bool)
            mask[idx] = True
            flat[row] *= mask
        configs.append(arr)
    return configs


CONFIGS = generate_cases()


def validate_nm(tensor, n=2, m=4):
    arr = np.asarray(tensor)
    orig_shape = arr.shape
    flat = arr.reshape(-1, m)
    nonzero_counts = np.sum(flat != 0, axis=1)
    valid = np.all(nonzero_counts <= n)
    return bool(valid), nonzero_counts.tolist()


def check_mask(mask, n=2, m=4):
    arr = np.asarray(mask, dtype=bool)
    flat = arr.reshape(-1, m)
    counts = np.sum(flat, axis=1)
    valid = np.all(counts <= n)
    return bool(valid), counts.tolist()


def pack_mask(mask):
    arr = np.asarray(mask, dtype=bool).flatten()
    packed = []
    for i in range(0, len(arr), 4):
        chunk = arr[i : i + 4]
        val = 0
        for bit_idx, b in enumerate(chunk):
            if b:
                val |= 1 << (bit_idx * 2)
        packed.append(val)
    return packed
