import numpy as np

FP4_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def fp4_quantize_scalar(val):
    """Quantize scalar or array values to nearest FP4 E2M1 code."""
    val_arr = np.asarray(val, dtype=np.float32)
    signs = np.sign(val_arr)
    abs_vals = np.abs(val_arr)
    diffs = np.abs(abs_vals[..., None] - FP4_VALUES)
    idx = np.argmin(diffs, axis=-1)
    q_abs = FP4_VALUES[idx]
    res = signs * q_abs
    if np.isscalar(val):
        return float(res)
    return res.astype(np.float32)


def nvfp4_quantize_dequantize(x, block_size=16, super_block_size=256):
    """Quantize and dequantize tensor using NVFP4 two-level scaling."""
    x_arr = np.asarray(x, dtype=np.float32)
    orig_shape = x_arr.shape
    flat_x = x_arr.ravel()
    n_total = flat_x.size

    if n_total % super_block_size != 0:
        raise ValueError("Input length must be a multiple of super_block_size")

    num_sb = n_total // super_block_size
    k_blocks = super_block_size // block_size

    x_3d = flat_x.reshape(num_sb, k_blocks, block_size)

    max_sb = np.max(np.abs(x_3d), axis=(1, 2))
    s2 = np.where(max_sb > 0, max_sb / (255.0 * 6.0), 1.0).astype(np.float32)

    max_b = np.max(np.abs(x_3d), axis=2)
    s2_expanded = s2[:, None]
    target_s1 = np.where(s2_expanded > 0, max_b / (s2_expanded * 6.0), 0.0)
    s1 = np.clip(np.round(target_s1), 0, 255).astype(np.uint8)

    S = (s1.astype(np.float32) * s2_expanded)[:, :, None]

    v = np.where(S > 0, x_3d / S, 0.0)

    diffs = np.abs(np.abs(v)[..., None] - FP4_VALUES)
    idx = np.argmin(diffs, axis=-1)
    q_abs = FP4_VALUES[idx]
    q = np.sign(v) * q_abs

    x_hat_3d = q * S
    x_hat = x_hat_3d.reshape(orig_shape).astype(np.float32)

    return x_hat, s1.ravel(), s2.ravel()


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
