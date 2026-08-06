import numpy as np


def compute_packing_sizes(num_weights, bits):
    total_bits = num_weights * bits
    bytes_unaligned = int(np.ceil(total_bits / 8.0))
    bytes_aligned = int(np.ceil(bytes_unaligned / 128.0) * 128)
    return {"unaligned_bytes": bytes_unaligned, "aligned_bytes": bytes_aligned}
