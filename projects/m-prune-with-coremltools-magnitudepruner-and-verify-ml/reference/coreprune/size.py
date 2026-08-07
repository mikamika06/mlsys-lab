import numpy as np


def derive_expected_size(shape, sparsity=0.5, n_bits=4, base_bytes=None):
    total_elements = int(np.prod(shape))
    if base_bytes is None:
        base_bytes = total_elements * 4
    active_elements = int(total_elements * (1.0 - sparsity))
    data_bytes = int(np.ceil(active_elements * n_bits / 8.0))
    palette_bytes = (2 ** n_bits) * 4
    overhead = int(total_elements * 0.05)
    expected = data_bytes + palette_bytes + overhead
    return expected


def verify_size_reduction(original_bytes, new_bytes, expected_ratio=0.5):
    ratio = new_bytes / float(original_bytes)
    return ratio <= expected_ratio, ratio
