import numpy as np
from q4k.quant import quantize_q4_k_superblock, dequantize_q4_k_superblock
from q4k.analysis import find_worst_subblocks


def test_q4k_byte_exact_roundtrip():
    np.random.seed(42)
    w = np.random.uniform(-1.0, 1.0, 256).astype(np.float32)
    b = quantize_q4_k_superblock(w)
    dec = dequantize_q4_k_superblock(b)
    b2 = quantize_q4_k_superblock(dec)
    assert b == b2


def test_subblock_ranking_length():
    np.random.seed(123)
    w = np.random.uniform(-0.5, 0.5, 256).astype(np.float32)
    b = quantize_q4_k_superblock(w)
    ranking = find_worst_subblocks(w, b)
    assert len(ranking) == 16
    assert len(set(ranking)) == 16
