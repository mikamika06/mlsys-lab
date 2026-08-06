import numpy as np


def fp32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    x_f32 = np.asarray(x, dtype=np.float32)
    u = x_f32.view(np.uint32)
    nan_mask = np.isnan(x_f32)

    lsb = (u >> 16) & 1
    bias = np.uint32(0x7FFF) + lsb
    rounded = ((u.astype(np.uint64) + bias.astype(np.uint64)) >> 16).astype(np.uint16)

    if np.any(nan_mask):
        nan_bits = ((u[nan_mask] >> 16) | np.uint32(0x0040)).astype(np.uint16)
        rounded[nan_mask] = nan_bits

    return rounded


def bf16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    bits_u16 = np.asarray(bits, dtype=np.uint16)
    u32 = bits_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def round_fp32_to_bf16(x: np.ndarray) -> np.ndarray:
    bits = fp32_to_bf16_bits(x)
    return bf16_bits_to_fp32(bits)


def fp16_subnormal_mask(x: np.ndarray) -> np.ndarray:
    arr16 = np.asarray(x, dtype=np.float16)
    u16 = arr16.view(np.uint16)
    exp = (u16 >> 10) & 0x1F
    mant = u16 & 0x03FF
    return (exp == 0) & (mant != 0)


def round_fp32_to_fp16(x: np.ndarray, flush_subnormals: bool = False) -> np.ndarray:
    x_f32 = np.asarray(x, dtype=np.float32)
    res = x_f32.astype(np.float16)
    if flush_subnormals:
        sub_mask = fp16_subnormal_mask(res)
        res[sub_mask] = np.copysign(0.0, res[sub_mask])
    return res


def get_dtype_max(dtype_str: str) -> float:
    d = dtype_str.lower()
    if d in ("fp32", "float32"):
        return float(np.finfo(np.float32).max)
    if d in ("fp16", "float16"):
        return float(np.finfo(np.float16).max)
    if d in ("bf16", "bfloat16"):
        return float((2.0 - 2.0**-7) * (2.0**127))
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def compute_ulp(x: np.ndarray, dtype_str: str) -> np.ndarray:
    d = dtype_str.lower()
    if d in ("fp32", "float32"):
        m, min_exp = 23, -126
    elif d in ("fp16", "float16"):
        m, min_exp = 10, -14
    elif d in ("bf16", "bfloat16"):
        m, min_exp = 7, -126
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    arr = np.asarray(np.abs(x), dtype=np.float64)
    res = np.zeros_like(arr, dtype=np.float64)

    non_zero = arr > 0
    if np.any(non_zero):
        vals = arr[non_zero]
        exp = np.floor(np.log2(vals))
        exp = np.maximum(exp, min_exp)
        res[non_zero] = 2.0 ** (exp - m)

    res[~non_zero] = 2.0 ** (min_exp - m)
    return res


def make_m1_dataset(seed=42):
    rng = np.random.default_rng(seed)
    normals = rng.uniform(-1000.0, 1000.0, size=500).astype(np.float32)
    subnormals = rng.uniform(-1e-38, 1e-38, size=200).astype(np.float32)
    ties = np.array([
        1.0,
        1.0 + 2.0**-17,
        1.0 + 2.0**-16,
        1.0 + 2.0**-16 + 2.0**-17,
        -1.0 - 2.0**-17,
        0.0,
        -0.0,
        np.nan,
        np.inf,
        -np.inf,
    ], dtype=np.float32)
    return np.concatenate([normals, subnormals, ties])


def make_m2_dataset(seed=1337):
    rng = np.random.default_rng(seed)
    vals = rng.uniform(-100.0, 100.0, size=300).astype(np.float32)
    subs = rng.uniform(-6e-5, 6e-5, size=200).astype(np.float32)
    zeros = np.array([0.0, -0.0], dtype=np.float32)
    return np.concatenate([vals, subs, zeros])
