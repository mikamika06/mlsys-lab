import numpy as np
from bnb_nf4.unpack import NF4_CODEBOOK, unpack_nibbles


def dequantize_nf4(packed: np.ndarray, absmax: np.ndarray, blocksize: int = 64) -> np.ndarray:
    codes = unpack_nibbles(packed)
    num_elements = codes.size
    dequantized = np.empty(num_elements, dtype=np.float32)
    num_blocks = (num_elements + blocksize - 1) // blocksize
    for i in range(num_blocks):
        start = i * blocksize
        end = min(start + blocksize, num_elements)
        dequantized[start:end] = NF4_CODEBOOK[codes[start:end]] * absmax[i]
    return dequantized
