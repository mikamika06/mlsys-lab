import numpy as np


def _format_limits(exponent_bits: int, mantissa_bits: int):
    bias = 2 ** (exponent_bits - 1) - 1
    max_exp = (2 ** exponent_bits - 2) - bias
    min_normal_exp = 1 - bias
    max_finite = (2.0 - 2.0 ** -mantissa_bits) * (2.0 ** max_exp)
    min_subnormal = (2.0 ** min_normal_exp) * (2.0 ** -mantissa_bits)
    return max_finite, min_subnormal


_FP16_MAX = float(np.finfo(np.float16).max)
_FP16_MIN_SUB = float(np.finfo(np.float16).smallest_subnormal)
_BF16_MAX, _BF16_MIN_SUB = _format_limits(exponent_bits=8, mantissa_bits=7)


def _classify(x, max_finite, min_subnormal):
    ax = abs(x)
    if ax > max_finite:
        return "overflow"
    if ax != 0.0 and ax < min_subnormal:
        return "underflow"
    return "ok"


def classify_fp_value(x: float) -> dict:
    """
    Classify how the real value x would be stored in fp16 and in bf16:
    'overflow' if |x| exceeds the format's largest finite magnitude,
    'underflow' if x is nonzero but smaller than the format's smallest
    positive subnormal (flushes to zero), else 'ok'.
    """
    return {
        "fp16": _classify(x, _FP16_MAX, _FP16_MIN_SUB),
        "bf16": _classify(x, _BF16_MAX, _BF16_MIN_SUB),
    }
