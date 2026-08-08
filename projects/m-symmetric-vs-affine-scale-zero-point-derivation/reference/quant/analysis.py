import numpy as np


def blockwise_size_ratio(shape, block_size, bits=4):
    total_elements = int(np.prod(shape))
    num_blocks = (total_elements + block_size - 1) // block_size
    weight_bits = total_elements * bits
    overhead_bits = num_blocks * 32
    compressed_bits = weight_bits + overhead_bits
    uncompressed_bits = total_elements * 16
    return float(compressed_bits / uncompressed_bits)
