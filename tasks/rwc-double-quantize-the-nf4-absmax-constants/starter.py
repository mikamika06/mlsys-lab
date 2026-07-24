import numpy as np


def nf4_double_quant_dequant(weights: np.ndarray, block_size: int, outer_block: int):
    """
    QLoRA-style double quantization.

    weights: array of any shape.
    block_size: level-1 NF4 block size (number of weight elements sharing
        one fp32 absmax constant c1).
    outer_block: number of consecutive c1 values grouped together for
        the level-2 8-bit (asymmetric, min-max) blockwise quantization of
        the absmax constants themselves.

    Returns (reconstructed_weights, bits_per_param) where
    bits_per_param = 4 + 8/block_size + 32/(block_size*outer_block).
    """
    raise NotImplementedError('your code here')
