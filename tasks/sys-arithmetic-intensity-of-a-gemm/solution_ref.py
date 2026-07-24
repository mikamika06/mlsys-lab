import numpy as np

def arithmetic_intensity(m, k, n, dtype):
    a = np.zeros((m, k), dtype=dtype)
    b = np.zeros((k, n), dtype=dtype)
    bytes_read = a.nbytes + b.nbytes
    c_bytes = (m * n) * a.itemsize
    total_bytes = bytes_read + c_bytes
    flops = 2 * m * k * n
    return flops / total_bytes
