import numpy as np

NF4_CODEBOOK = np.array([
    -1.0,
    -0.6961928009986877,
    -0.5250929000305176,
    -0.39491748809814453,
    -0.28444117307662964,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791837096214294,
    0.4407098293384552,
    0.5626170039176941,
    0.722956836227417,
    1.0,
], dtype=np.float64)


def quantize_nf4(x, block_size=64):
    raise NotImplementedError


def dequantize_nf4(q_indices, scales, orig_shape, orig_len):
    raise NotImplementedError


def measure_nf4_cycle_error(w_init, num_cycles, update_fn=None, block_size=64):
    raise NotImplementedError
