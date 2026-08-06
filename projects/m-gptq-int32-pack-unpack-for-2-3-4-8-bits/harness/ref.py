import numpy as np
from reference.quantpack import pack_weights, unpack_weights, convert_awq_to_gptq
from reference.quantpack.layout import get_packed_shape, get_memory_strides

TEST_CASES = [
    (np.array([1, 2, 3, 0], dtype=np.int32), 2),
    (np.array([5, 1, 7, 2, 0, 3, 4, 1], dtype=np.int32), 4),
    (np.array([120, 10, 5, 255], dtype=np.int32), 8),
    (np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32), 3),
]

LAYOUT_CASES = [
    (16, 32, 2),
    (32, 64, 4),
    (64, 128, 8),
    (16, 30, 3),
]
