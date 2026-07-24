import numpy as np


def _e4m3_table():
    codes = np.arange(256, dtype=np.uint8)
    sign = ((codes >> 7) & 1).astype(np.float64)
    exp = ((codes >> 3) & 0xF).astype(np.float64)
    mant = (codes & 0x7).astype(np.float64)
    s = 1.0 - 2.0 * sign
    subnormal = exp == 0
    normal_val = s * (2.0 ** (exp - 7.0)) * (1.0 + mant / 8.0)
    subnormal_val = s * (2.0 ** (-6.0)) * (mant / 8.0)
    val = np.where(subnormal, subnormal_val, normal_val)
    nan_mask = (exp == 15) & (mant == 7)
    val = np.where(nan_mask, np.nan, val)
    return val


_TABLE = _e4m3_table()
_FINITE_MASK = ~np.isnan(_TABLE)
_FINITE_VALS = _TABLE[_FINITE_MASK]
_FINITE_CODES = np.arange(256, dtype=np.uint8)[_FINITE_MASK]


def _e4m3_encode(x):
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    diff = np.abs(flat[:, None] - _FINITE_VALS[None, :])
    idx = np.argmin(diff, axis=1)
    return _FINITE_CODES[idx].reshape(x.shape).astype(np.uint8)


def _e4m3_decode(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    return _TABLE[codes]


def _to_bf16(x):
    x32 = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    bits = x32.view(np.uint32)
    rounding_bias = ((bits >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    bits_rounded = (bits + rounding_bias) & np.uint32(0xFFFF0000)
    return bits_rounded.view(np.float32).reshape(x32.shape)


def cast_and_matmul_fp8e4m3(W: np.ndarray, X: np.ndarray):
    """
    Store W in E4M3 (nearest-value 8-bit codes), upcast the decoded grid
    value to bfloat16 precision for compute, snap X to bfloat16 too, and
    matmul with float32 accumulation. Returns (Y, codes).
    """
    codes = _e4m3_encode(W)
    w_fp8 = _e4m3_decode(codes)
    w_bf16 = _to_bf16(w_fp8)
    x_bf16 = _to_bf16(X)
    Y = x_bf16.astype(np.float32) @ w_bf16.astype(np.float32)
    return Y, codes
