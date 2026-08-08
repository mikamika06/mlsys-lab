import numpy as np

def packed_shape_and_stride(shape, bits):
    rows, cols = shape
    values_per_int = 32 // bits
    if bits == 3:
        packed_rows = (rows * cols + 9) // 10
        return (packed_rows, 1), (1, 1)
    else:
        packed_rows = (rows * cols + values_per_int - 1) // values_per_int
        return (packed_rows, 1), (1, 1)
