import numpy as np


def _midrange_probe_values(mantissa_bits: int) -> np.ndarray:
    """Worst-case-adjacent probe values for a `mantissa_bits`-mantissa
    format: for several exponents in a "mid-range" band (well away from
    subnormal/overflow edges), place a value exactly halfway between the
    two nearest representable mantissa grid points closest to the binade's
    low end -- this is where round-to-nearest relative error is largest.
    """
    exps = np.arange(-3, 5, dtype=np.float64)
    q = 2.0 ** -mantissa_bits
    f_mid = q / 2.0  # halfway between mantissa grid points 0 and q
    vals = (2.0 ** exps) * (1.0 + f_mid)
    return np.concatenate([vals, -vals])


def _oracle_round(values: np.ndarray, mantissa_bits: int) -> np.ndarray:
    """Round each value to `mantissa_bits` mantissa bits (round-to-nearest,
    unbounded exponent -- pure mantissa-quantization model), via real
    floating-point mantissa extraction (np.frexp), not a hardcoded table.
    """
    v = np.asarray(values, dtype=np.float64)
    sign = np.sign(v)
    av = np.abs(v)
    m, e = np.frexp(av)          # av = m * 2**e,  m in [0.5, 1)
    m2, e2 = m * 2.0, e - 1      # av = m2 * 2**e2, m2 in [1, 2)
    f = m2 - 1.0
    q = 2.0 ** -mantissa_bits
    f_r = np.round(f / q) * q
    return sign * (1.0 + f_r) * (2.0 ** e2)


def _oracle_step(mantissa_bits: int):
    values = _midrange_probe_values(mantissa_bits)
    analytic = 2.0 ** -(mantissa_bits + 1)
    rounded = _oracle_round(values, mantissa_bits)
    empirical = float(np.max(np.abs(rounded - values) / np.abs(values)))
    return values, analytic, empirical


def grade(sol, fx) -> dict:
    # e4m3's own mantissa width (3 bits) is the headline case; a couple of
    # neighboring mantissa widths are included so a solution can't just
    # memorize the single number 0.0625 -- it must implement the general
    # rounding-step derivation.
    worst_rel_err = 0.0

    for mantissa_bits in (3, 4, 5):
        values, analytic_ref, empirical_ref = _oracle_step(mantissa_bits)

        try:
            got = sol.relative_rounding_step(values.copy(), mantissa_bits)
        except Exception:
            return {"rel_err": float("inf")}

        try:
            analytic_got, empirical_got = got
            analytic_got = float(analytic_got)
            empirical_got = float(empirical_got)
        except Exception:
            return {"rel_err": float("inf")}

        # Sanity: both returned numbers must actually be the real ones,
        # not just mutually-consistent placeholders.
        if not np.isclose(analytic_got, analytic_ref, rtol=1e-9, atol=0.0):
            return {"rel_err": float("inf")}
        if not np.isclose(empirical_got, empirical_ref, rtol=1e-6, atol=1e-12):
            return {"rel_err": float("inf")}

        rel_err = abs(empirical_got - analytic_got) / abs(analytic_got)
        worst_rel_err = max(worst_rel_err, rel_err)

    return {"rel_err": worst_rel_err}
