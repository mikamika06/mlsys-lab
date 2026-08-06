import numpy as np


def double_quant_bits_per_param(params_count, block_size=64, outer_block_size=256):
    base_bits = 4.0
    num_blocks = np.ceil(params_count / block_size)
    outer_blocks = np.ceil(num_blocks / outer_block_size)
    absmax_bits_double = (32.0 * num_blocks + 8.0 * num_blocks + 32.0 * outer_blocks) / params_count
    return base_bits + absmax_bits_double
