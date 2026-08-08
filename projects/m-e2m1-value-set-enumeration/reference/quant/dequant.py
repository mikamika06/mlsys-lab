import numpy as np
from quant.e2m1 import enumerate_e2m1_values


def dequantize_block(packed_bytes, scale):
    table = enumerate_e2m1_values()
    nibbles = []
    for b in packed_bytes:
        high = (b >> 4) & 0xF
        low = b & 0xF
        nibbles.append(high)
        nibbles.append(low)
    vals = table[np.array(nibbles, dtype=int)]
    return vals * float(scale)
