import numpy as np


def q4k_quantize_superblock(x):
    """Q4_K two-level (super-block) asymmetric quantization.

    x: float32 array (rows, cols), cols a multiple of 256. Each row splits
    into super-blocks of 256 values, each super-block into 8 sub-blocks of
    32 values.

    Per sub-block i: mn_i, mx_i = min/max(sub_i); ss_i = (mx_i-mn_i)/63;
    mm_i = -mn_i/63. Per super-block: d = max_i(ss_i), dmin = max_i(mm_i).
    Per sub-block 6-bit codes: sc_i = round(ss_i/d*63), mc_i =
    round(mm_i/dmin*63), both clipped to [0, 63]. Per weight:
    step = d*sc_i, off = dmin*mc_i, q = clip(round((w+off)/step), 0, 15).

    Returns (codes, sub_scales, sub_mins, d, dmin):
      codes      -- uint8, shape (rows, cols // 2), 4-bit codes packed two
                    per byte (low nibble = even index, high nibble = odd).
      sub_scales -- uint8, shape (rows, cols // 256, 8), 6-bit codes.
      sub_mins   -- uint8, shape (rows, cols // 256, 8), 6-bit codes.
      d          -- float16, shape (rows, cols // 256).
      dmin       -- float16, shape (rows, cols // 256).
    """
    raise NotImplementedError('your code here')


def q4k_dequantize_superblock(codes, sub_scales, sub_mins, d, dmin):
    """Inverse of `q4k_quantize_superblock`.

    w_hat = d[sb] * sub_scales[sb, i] * q - dmin[sb] * sub_mins[sb, i]
    for every sub-block i (32 values) inside every super-block sb.

    Returns a float32 array of shape (rows, n_super_blocks * 256).
    """
    raise NotImplementedError('your code here')
