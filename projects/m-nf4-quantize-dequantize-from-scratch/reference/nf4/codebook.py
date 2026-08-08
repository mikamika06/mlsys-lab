import numpy as np


def _erfinv(x: np.ndarray) -> np.ndarray:
    a = 0.147
    log_term = np.log(1.0 - x**2)
    term1 = 2.0 / (np.pi * a) + log_term / 2.0
    inner = term1**2 - log_term / a
    sign = np.sign(x)
    return sign * np.sqrt(np.sqrt(inner) - term1)


def generate_nf4_codebook() -> np.ndarray:
    """Generate the 16-element NF4 quantile codebook normalized to [-1, 1]."""
    pos_q = np.linspace(0.5, 1.0, 9)
    neg_q = np.linspace(0.0, 0.5, 9)

    pos_v = np.sqrt(2.0) * _erfinv(2.0 * pos_q - 1.0)
    neg_v = np.sqrt(2.0) * _erfinv(2.0 * neg_q - 1.0)

    pos_norm = pos_v / pos_v[-1]
    neg_norm = neg_v / np.abs(neg_v[0])

    cb = np.zeros(16, dtype=np.float64)
    cb[:8] = neg_norm[:8]
    cb[8:] = pos_norm[1:]
    return cb


def generate_fp4_codebook() -> np.ndarray:
    """Generate FP4 (E2M1) normalized codebook sorted in ascending order."""
    vals = [0.0, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0]
    full = []
    for v in vals:
        if v != 0.0:
            full.append(-v)
        full.append(v)
    cb = np.array(sorted(full), dtype=np.float64)
    return cb / np.max(cb)


def generate_int4_codebook() -> np.ndarray:
    """Generate symmetric 4-bit integer codebook mapped to [-1, 1]."""
    vals = np.arange(-7, 8, dtype=np.float64)
    cb = np.zeros(16, dtype=np.float64)
    cb[0] = -1.0
    cb[1:] = vals / 7.0
    return cb
