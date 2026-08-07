import numpy as np
from .quant import tensor_byte_size


def quantize_tensor(tensor_data, target_type):
    arr = np.asarray(tensor_data, dtype=np.float32)
    dims = list(arr.shape)
    n_el = int(np.prod(dims))
    size = tensor_byte_size(dims, target_type)
    return bytearray(size)
