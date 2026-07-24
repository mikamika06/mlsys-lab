import numpy as np


def _oracle_qparams(x, nbits):
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << nbits) - 1
    mn = min(0.0, float(np.min(x)))
    mx = max(0.0, float(np.max(x)))
    if mx == mn:
        scale = 1.0
    else:
        scale = (mx - mn) / qmax
    zp = int(np.clip(round(-mn / scale), 0, qmax))
    return float(scale), zp


def _reconstruct(x, scale, zp, nbits):
    qmax = (1 << nbits) - 1
    codes = np.clip(np.round(x / scale) + zp, 0, qmax)
    return (codes - zp) * scale


def grade(sol, fx) -> dict:
    """
    Re-derives the optimal asymmetric (min-max, zero-including) scale and
    zero-point with a NumPy oracle, on several random groups + a degenerate
    constant group, and compares the two reconstructions' max-abs difference.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    trials = []
    for _ in range(7):
        n = int(rng.integers(3, 40))
        nbits = int(rng.choice([2, 3, 4, 8]))
        x = rng.normal(size=n) * rng.uniform(0.1, 50.0) + rng.uniform(-20.0, 20.0)
        trials.append((x, nbits))
    # one degenerate constant-array trial
    trials.append((np.full(5, 3.5), 4))
    trials.append((np.zeros(6), 3))

    for x, nbits in trials:
        try:
            scale_exp, zp_exp = _oracle_qparams(x, nbits)
            got = sol.derive_affine_qparams(np.asarray(x).copy(), nbits)
            scale_got, zp_got = got
            scale_got = float(scale_got)
            zp_got = int(zp_got)

            recon_exp = _reconstruct(x, scale_exp, zp_exp, nbits)
            recon_got = _reconstruct(x, scale_got, zp_got, nbits)
            err = float(np.max(np.abs(recon_got - recon_exp)))
        except Exception:
            err = float("inf")
        worst = max(worst, err)

    return {"max_abs_err": worst}
