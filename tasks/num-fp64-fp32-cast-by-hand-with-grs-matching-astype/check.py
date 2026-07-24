import numpy as np
from mlsys import scorers


def _make_double(sign: int, mant52_bits: int, exp_field: int = 1023) -> float:
    bits = (
        (np.uint64(sign) << np.uint64(63))
        | (np.uint64(exp_field) << np.uint64(52))
        | np.uint64(mant52_bits)
    )
    return np.array([bits], dtype=np.uint64).view(np.float64)[0]


def _inputs() -> np.ndarray:
    rng = np.random.default_rng(0)

    n = 20000
    mant = rng.uniform(1.0, 2.0, size=n)
    exp = rng.integers(-100, 100, size=n)
    sign = rng.choice([1.0, -1.0], size=n)
    bulk = (sign * mant * (2.0 ** exp)).astype(np.float64)

    # exact tie, kept mantissa already even -> round DOWN
    tie_even = _make_double(0, 1 << 28)
    # exact tie, kept mantissa odd -> round UP to the even neighbour
    tie_odd = _make_double(0, (1 << 29) | (1 << 28))
    # kept mantissa all-ones + a definite round-up -> overflow, carry into exponent
    carry = _make_double(0, (0x7FFFFF << 29) | (1 << 28) | 1)
    # same three, negated, and shifted to a different exponent field
    tie_even_neg = _make_double(1, 1 << 28, exp_field=1050)
    tie_odd_hi = _make_double(0, (1 << 29) | (1 << 28), exp_field=980)
    carry_hi = _make_double(0, (0x7FFFFF << 29) | (1 << 28) | 1, exp_field=1000)

    edge = np.array(
        [tie_even, tie_odd, carry, tie_even_neg, tie_odd_hi, carry_hi],
        dtype=np.float64,
    )
    return np.concatenate([bulk, edge])


def grade(sol, fx) -> dict:
    x = _inputs()
    ref = x.astype(np.float32).view(np.uint32)

    try:
        got = np.asarray(sol.fp64_to_fp32_bits(x))
    except Exception:
        return {"byte_exact_fraction": 0.0}

    if got.shape != x.shape or got.dtype != np.uint32:
        return {"byte_exact_fraction": 0.0}

    return {"byte_exact_fraction": scorers.byte_exact_fraction(got, ref)}
