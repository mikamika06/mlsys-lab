import numpy as np
from mlsys.scorers import mean_kl

def grade(sol, fx) -> dict:
    # deterministic random data
    rng = np.random.default_rng(0)
    base = rng.standard_normal((12, 20))
    pi   = rng.standard_normal((12, 20))
    ntk  = rng.standard_normal((12, 20))

    try:
        got_pi, got_ntk = sol.compare_pi_ntk(base.tolist(), pi.tolist(), ntk.tolist())
    except Exception as e:
        return {"mean_kl_pi": 0.0, "mean_kl_ntk": 0.0}

    # reference values
    ref_pi = mean_kl(base, pi)
    ref_ntk = mean_kl(base, ntk)

    ok_pi = float(np.isclose(got_pi, ref_pi, rtol=1e-9, atol=0.0))
    ok_ntk = float(np.isclose(got_ntk, ref_ntk, rtol=1e-9, atol=0.0))

    return {"mean_kl_pi": ok_pi, "mean_kl_ntk": ok_ntk}
