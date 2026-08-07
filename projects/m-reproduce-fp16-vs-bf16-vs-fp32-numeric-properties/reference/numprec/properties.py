import numpy as np

FORMAT_SPECS = {
    "fp16": {
        "exponent_bits": 5,
        "mantissa_bits": 10,
        "max_val": 65504.0,
        "min_pos_normal": 2.0**(-14),
        "min_pos_subnormal": 2.0**(-24),
        "eps": 2.0**(-10),
    },
    "bf16": {
        "exponent_bits": 8,
        "mantissa_bits": 7,
        "max_val": (2.0 - 2.0**(-7)) * (2.0**127),
        "min_pos_normal": 2.0**(-126),
        "min_pos_subnormal": 2.0**(-133),
        "eps": 2.0**(-7),
    },
    "fp32": {
        "exponent_bits": 8,
        "mantissa_bits": 23,
        "max_val": (2.0 - 2.0**(-23)) * (2.0**127),
        "min_pos_normal": 2.0**(-126),
        "min_pos_subnormal": 2.0**(-149),
        "eps": 2.0**(-23),
    },
}


def get_format_properties(dtype_str: str) -> dict:
    if dtype_str not in FORMAT_SPECS:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    return dict(FORMAT_SPECS[dtype_str])


def _cast_fp16(x: float) -> float:
    return float(np.float16(x))


def _cast_bf16(x: float) -> float:
    u32 = np.frombuffer(np.float32(x).tobytes(), dtype=np.uint32)[0]
    rounding_bias = 0x00007FFF + ((u32 >> 16) & 1)
    u32_rounded = u32 + rounding_bias
    bf16_u16 = np.uint16(u32_rounded >> 16)
    bf32_u32 = np.uint32(bf16_u16) << 16
    return float(np.frombuffer(bf32_u32.tobytes(), dtype=np.float32)[0])


def _cast_fp32(x: float) -> float:
    return float(np.float32(x))


def compute_relative_error(val: float, dtype_str: str) -> float:
    if val == 0.0:
        return 0.0
    if dtype_str == "fp16":
        casted = _cast_fp16(val)
    elif dtype_str == "bf16":
        casted = _cast_bf16(val)
    elif dtype_str == "fp32":
        casted = _cast_fp32(val)
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    return abs(val - casted) / abs(val)
