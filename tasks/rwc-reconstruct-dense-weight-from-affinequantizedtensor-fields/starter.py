import numpy as np


def reconstruct_dense_from_affine_quantized(int_data_packed, scale, zero_point, group_size, shape):
    """Rebuild the dense dequantized weight from AffineQuantizedTensor-style
    storage: nibble-packed int4 codes plus per-group (scale, zero_point).

    int_data_packed -- uint8 array of length ceil(n/2), n = prod(shape).
        Byte i packs two 4-bit codes for the raveled (row-major) elements
        2*i (low nibble) and 2*i+1 (high nibble, unused/0 if 2*i+1 >= n).
    scale, zero_point -- float64 arrays of length ceil(n/group_size), one
        pair per group of `group_size` consecutive raveled elements.
    group_size -- int.
    shape -- target dense shape.

    Dequant per element: (code - zero_point[group]) * scale[group].
    Returns a float64 array of `shape`.
    """
    raise NotImplementedError('your code here')
