import numpy as np
import mpmath as mp

from mlsys import scorers


def _mp_logsumexp(x) -> float:
    """Real high-precision oracle: 50 decimal digits via mpmath, immune to
    float64 overflow/underflow, so it stays accurate on the adversarial
    fixtures too."""
    with mp.workdps(50):
        s = mp.mpf(0)
        for xi in x:
            s += mp.e ** mp.mpf(float(xi))
        return float(mp.log(s))


def grade(sol, fx) -> dict:
    cases = [
        np.asarray(fx["x_overflow"], dtype=np.float64),
        np.asarray(fx["x_underflow"], dtype=np.float64),
        np.array([2000.0, -2000.0, 1999.3, 1998.1], dtype=np.float64),  # mixed extreme range
    ]

    rng = np.random.default_rng(0)
    for _ in range(4):
        n = int(rng.integers(3, 12))
        cases.append(rng.uniform(-5.0, 5.0, size=n))  # ordinary, non-extreme range

    ref_vals = []
    got_vals = []
    for x in cases:
        ref = _mp_logsumexp(x)
        try:
            got = sol.logsumexp_stable(x)
            got = float(got)
        except Exception:
            return {"rel_err": float("inf")}

        if not np.isfinite(got):
            return {"rel_err": float("inf")}

        ref_vals.append(ref)
        got_vals.append(got)

    return {"rel_err": scorers.rel_err(np.array(ref_vals), np.array(got_vals))}
