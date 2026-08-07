import math


def relative_rounding_step(values: list[float], mantissa_bits: int) -> tuple[float, float]:
    """
    Derive the analytic max relative rounding error for a `mantissa_bits`
    -mantissa floating-point format, and measure it empirically on a
    mid-range probe list.

    Returns (analytic_bound, empirical_max_rel_err).
    """
    analytic_bound = 2.0 ** -(mantissa_bits + 1)
    q = 2.0 ** -mantissa_bits

    empirical_max_rel_err = 0.0
    for v_i in values:
        if v_i < 0.0:
            sign_i = -1.0
        elif v_i > 0.0:
            sign_i = 1.0
        else:
            sign_i = 0.0
        av_i = math.fabs(v_i)
        m, e = math.frexp(av_i)
        m2, e2 = m * 2.0, e - 1
        f = m2 - 1.0
        f_rounded = round(f / q) * q
        rounded_i = sign_i * (1.0 + f_rounded) * (2.0 ** e2)
        rel_err_i = math.fabs(rounded_i - v_i) / math.fabs(v_i)
        if rel_err_i > empirical_max_rel_err:
            empirical_max_rel_err = rel_err_i

    return analytic_bound, empirical_max_rel_err
