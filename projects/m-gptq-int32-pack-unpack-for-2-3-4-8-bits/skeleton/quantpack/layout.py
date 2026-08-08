import numpy as np

def get_packed_shape(rows: int, cols: int, bits: int) -> tuple:
    raise NotImplementedError

def get_memory_strides(shape: tuple, dtype_size: int) -> tuple:
    raise NotImplementedError
