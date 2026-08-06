import numpy as np


def encode_bitmask_values(tensor, block_size=8):
    flat = np.asarray(tensor, dtype=np.float32).flatten()
    n = len(flat)
    padded_len = ((n + block_size - 1) // block_size) * block_size
    if padded_len > n:
        flat = np.pad(flat, (0, padded_len - n), mode='constant')

    blocks = flat.reshape(-1, block_size)
    values_list = []
    bitmasks = []

    for block in blocks:
        nonzero_mask = block != 0.0
        mask_val = 0
        for i, present in enumerate(nonzero_mask):
            if present:
                mask_val |= (1 << i)
                values_list.append(block[i])
        bitmasks.append(mask_val)

    return {
        "bitmasks": np.array(bitmasks, dtype=np.uint64),
        "values": np.array(values_list, dtype=np.float32),
        "original_shape": tensor.shape if hasattr(tensor, "shape") else np.shape(tensor),
        "block_size": block_size,
        "original_size": n
    }


def decode_bitmask_values(encoded, shape, block_size=8):
    bitmasks = encoded["bitmasks"]
    values = encoded["values"]
    original_size = encoded.get("original_size", np.prod(shape))

    reconstructed = []
    val_idx = 0

    for mask in bitmasks:
        block = np.zeros(block_size, dtype=np.float32)
        for i in range(block_size):
            if (mask & (1 << i)):
                block[i] = values[val_idx]
                val_idx += 1
        reconstructed.append(block)

    flat = np.concatenate(reconstructed)[:original_size]
    return flat.reshape(shape)
