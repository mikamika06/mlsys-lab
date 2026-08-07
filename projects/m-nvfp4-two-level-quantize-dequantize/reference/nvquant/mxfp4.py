import numpy as np

FP4_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def mxfp4_quantize_dequantize(x, block_size=32):
    """Quantize and dequantize tensor using MXFP4 single-level scaling."""
    x_arr = np.asarray(x, dtype=np.float32)
    orig_shape = x_arr.shape
    flat_x = x_arr.ravel()
    n_total = flat_x.size

    if n_total % block_size != 0:
        raise ValueError("Input length must be a multiple of block_size")

    num_blocks = n_total // block_size
    x_2d = flat_x.reshape(num_blocks, block_size)

    max_b = np.max(np.abs(x_2d), axis=1)

    exp = np.ceil(np.log2(np.maximum(max_b / 6.0, 1e-12)))
    s = np.where(max_b > 0, 2.0 ** exp, 1.0).astype(np.float32)

    s_expanded = s[:, None]
    v = np.where(s_expanded > 0, x_2d / s_expanded, 0.0)

    diffs = np.abs(np.abs(v)[..., None] - FP4_VALUES)
    idx = np.argmin(diffs, axis=-1)
    q_abs = FP4_VALUES[idx]
    q = np.sign(v) * q_abs

    x_hat_2d = q * s_expanded
    x_hat = x_hat_2d.reshape(orig_shape).astype(np.float32)

    return x_hat, s.ravel()
