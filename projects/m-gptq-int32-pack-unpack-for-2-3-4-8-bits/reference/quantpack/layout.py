import numpy as np

def get_packed_shape(rows: int, cols: int, bits: int) -> tuple:
    vals = 32 // bits if bits != 3 else 10
    total_elements = rows * cols
    packed_len = (total_elements + vals - 1) // vals
    return (packed_len,)

def get_memory_strides(shape: tuple, dtype_size: int = 4) -> tuple:
    strides = []
    curr = dtype_size
    for dim in reversed(shape):
        strides.insert(0, curr)
        curr *= dim
    return tuple(strides)
