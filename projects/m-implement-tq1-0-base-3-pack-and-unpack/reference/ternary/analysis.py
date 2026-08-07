import numpy as np


def compute_iq4_xs_size(num_elements):
    block_size = 256
    num_blocks = (num_elements + block_size - 1) // block_size
    bytes_per_block = 136
    return num_blocks * bytes_per_block


def compute_q4_k_s_size(num_elements):
    block_size = 256
    num_blocks = (num_elements + block_size - 1) // block_size
    bytes_per_block = 144
    return num_blocks * bytes_per_block


def measure_imatrix_effect(codebook, imatrix_weights):
    cb = np.asarray(codebook, dtype=np.float32)
    iw = np.asarray(imatrix_weights, dtype=np.float32)
    if len(cb) != len(iw):
        iw = np.resize(iw, len(cb))
    weighted_variance = np.sum(iw * (cb ** 2)) / np.sum(iw)
    return float(weighted_variance)
