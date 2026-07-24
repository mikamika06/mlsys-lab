import numpy as np


def relative_rounding_step(values: np.ndarray, mantissa_bits: int):
    """Derive and empirically verify the max relative rounding error of a
    `mantissa_bits`-mantissa floating-point format.

    values: probe tensor of fp64 values known to sit in the format's
        normal (non-subnormal, non-overflowing) representable range.
    mantissa_bits: number of mantissa bits of the format being modeled
        (e4m3 has 3).

    Returns (analytic_bound, empirical_max_rel_err):
      - analytic_bound = 2 ** -(mantissa_bits + 1), the classic half-ULP
        max relative rounding error bound for round-to-nearest.
      - empirical_max_rel_err = max over `values` of
        |round_to_mantissa(v) - v| / |v|, where round_to_mantissa rounds
        v's normalized fractional mantissa (v = sign * (1+f) * 2**e,
        f in [0,1)) to the nearest multiple of 2 ** -mantissa_bits.
    """
    raise NotImplementedError('your code here')
