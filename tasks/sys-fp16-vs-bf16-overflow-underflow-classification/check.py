import numpy as np


def _format_limits(exponent_bits: int, mantissa_bits: int):
    """Max finite value and smallest positive subnormal for an IEEE-754-style
    binary format with the given exponent/mantissa bit widths, derived purely
    from the format definition (bias, exponent range, mantissa resolution)."""
    bias = 2 ** (exponent_bits - 1) - 1
    max_exp = (2 ** exponent_bits - 2) - bias          # top stored exponent reserved for inf/nan
    min_normal_exp = 1 - bias
    max_finite = (2.0 - 2.0 ** -mantissa_bits) * (2.0 ** max_exp)
    min_subnormal = (2.0 ** min_normal_exp) * (2.0 ** -mantissa_bits)
    return max_finite, min_subnormal


# fp16: pull the real thresholds straight from NumPy's actual float16 dtype.
_FP16_MAX = float(np.finfo(np.float16).max)
_FP16_MIN_SUB = float(np.finfo(np.float16).smallest_subnormal)

# bf16: no native NumPy dtype, so derive from the format definition
# (8 exponent bits, 7 mantissa bits, same exponent range as fp32).
_BF16_MAX, _BF16_MIN_SUB = _format_limits(exponent_bits=8, mantissa_bits=7)


def _classify(x: float, max_finite: float, min_subnormal: float) -> str:
    ax = abs(x)
    if ax > max_finite:
        return "overflow"
    if ax != 0.0 and ax < min_subnormal:
        return "underflow"
    return "ok"


def _oracle(x: float) -> dict:
    return {
        "fp16": _classify(x, _FP16_MAX, _FP16_MIN_SUB),
        "bf16": _classify(x, _BF16_MAX, _BF16_MIN_SUB),
    }


def _gen_values(rng):
    vals = [
        0.0, 1.0, -1.0,
        _FP16_MAX, -_FP16_MAX,
        _FP16_MIN_SUB, -_FP16_MIN_SUB,
        _BF16_MAX, -_BF16_MAX,
        _BF16_MIN_SUB, -_BF16_MIN_SUB,
        _FP16_MAX * 2.0,        # overflows fp16, fine in bf16
        _FP16_MIN_SUB / 2.0,    # underflows fp16, fine in bf16
        _BF16_MAX * 10.0,       # overflows both
        _BF16_MIN_SUB / 10.0,   # underflows both
    ]
    for _ in range(20):
        exp = rng.uniform(-46, 39)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        vals.append(sign * (10.0 ** exp))
    return vals


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    values = _gen_values(rng)

    ok = 1.0
    for x in values:
        expected = _oracle(x)
        try:
            got = sol.classify_fp_value(float(x))
            got_norm = {"fp16": str(got["fp16"]), "bf16": str(got["bf16"])}
        except Exception:
            ok = 0.0
            break
        if got_norm != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
