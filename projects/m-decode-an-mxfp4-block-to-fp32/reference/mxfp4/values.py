import numpy as np
from mxfp4.decode import decode_block


def enumerate_values(scale_byte):
    nibbles = np.arange(16, dtype=np.uint8)
    decoded = decode_block(scale_byte, nibbles)
    return sorted(list(set(decoded)))
