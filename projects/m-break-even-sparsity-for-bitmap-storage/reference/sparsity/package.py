import numpy as np


def measure_package_size(weights, mask, block_size):
    nz = np.sum(mask)
    num_elements = weights.size
    element_size = weights.dtype.itemsize
    bitmap_bytes = int(np.ceil(num_elements / 8))
    data_bytes = int(nz * element_size)
    index_bytes = int(nz * 2)
    return bitmap_bytes + data_bytes + index_bytes
