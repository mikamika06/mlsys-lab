import numpy as np
from sparsecoder.codec import encode_bitmask_values, decode_bitmask_values
from sparsecoder.analysis import breakeven_sparsity, measure_byte_savings

TENSORS = [
    np.array([[0.0, 1.2, 0.0, 0.0], [0.0, 0.0, -3.4, 0.0], [5.6, 0.0, 0.0, 0.0]], dtype=np.float32),
    np.zeros((16, 16), dtype=np.float32),
    np.ones((8, 8), dtype=np.float32)
]
