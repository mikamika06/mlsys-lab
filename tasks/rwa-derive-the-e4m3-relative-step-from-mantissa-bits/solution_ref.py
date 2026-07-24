import numpy as np


def relative_rounding_step(values: np.ndarray, mantissa_bits: int):
    """
    Derive the analytic max relative rounding error for a `mantissa_bits`
    -mantissa floating-point format, and measure it empirically on a
    mid-range probe tensor.

    Returns (analytic_bound, empirical_max_rel_err).
    """
    analytic_bound = 2.0 ** -(mantissa_bits + 1)

    v = np.asarray(values, dtype=np.float64)
    sign = np.sign(v)
    av = np.abs(v)

    # Extract the normalized binary mantissa: av = m2 * 2**e2, m2 in [1, 2).
    m, e = np.frexp(av)      # av = m * 2**e, m in [0.5, 1)
    m2, e2 = m * 2.0, e - 1
    f = m2 - 1.0              # fractional mantissa in [0, 1)

    # Round the fraction to the nearest multiple of 2**-mantissa_bits.
    q = 2.0 ** -mantissa_bits
    f_rounded = np.round(f / q) * q

    rounded = sign * (1.0 + f_rounded) * (2.0 ** e2)
    rel_errs = np.abs(rounded - v) / np.abs(v)
    empirical_max_rel_err = float(np.max(rel_errs))

    return analytic_bound, empirical_max_rel_err
