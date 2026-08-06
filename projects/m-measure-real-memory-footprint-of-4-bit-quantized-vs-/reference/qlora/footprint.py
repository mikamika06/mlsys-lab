import numpy as np


def measure_footprint(tensor, block_size=64):
    flat = tensor.flatten()
    numel = flat.size
    num_blocks = (numel + block_size - 1) // block_size
    weight_bytes = (numel * 4 + 1) // 2
    scale_bytes = num_blocks * 4
    total_bytes = weight_bytes + scale_bytes
    return {
        "numel": numel,
        "num_blocks": num_blocks,
        "weight_bytes": weight_bytes,
        "scale_bytes": scale_bytes,
        "total_bytes": total_bytes,
        "original_bytes": numel * np.dtype(tensor.dtype).itemsize,
    }
