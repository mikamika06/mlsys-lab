"""Codebook derivations for NF4, FP4, and INT4 formats."""

import numpy as np


def norm_ppf(p):
    """Acklam's approximation for standard normal quantile function."""
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def create_nf4_codebook():
    """Derive 16-element NF4 codebook from normal distribution quantiles."""
    q = np.zeros(16, dtype=np.float64)
    for i in range(7):
        p = (2 * i + 1) / 28.0
        q[i] = norm_ppf(p)
    q[7] = 0.0
    for i in range(8):
        p = 0.5 + (2 * i + 1) / 32.0
        q[i + 8] = norm_ppf(p)
    q = q / np.max(np.abs(q))
    return q


def create_fp4_codebook():
    """Derive 16-element FP4 (E2M1) normalized codebook."""
    vals = []
    for e in [0, 1, 2, 3]:
        for m in [0, 1]:
            if e == 0:
                val = -(m / 2.0)
            else:
                val = -(2.0 ** (e - 1)) * (1.0 + m / 2.0)
            vals.append(val)
    for e in [0, 1, 2, 3]:
        for m in [0, 1]:
            if e == 0:
                val = m / 2.0
            else:
                val = (2.0 ** (e - 1)) * (1.0 + m / 2.0)
            vals.append(val)
    vals = np.array(vals, dtype=np.float64)
    max_val = np.max(np.abs(vals))
    vals = np.sort(vals / max_val)
    return vals


def create_int4_codebook():
    """Derive 16-element uniform INT4 codebook in [-1, 1]."""
    return np.linspace(-1.0, 1.0, 16, dtype=np.float64)
