import numpy as np
from mlsys.scorers import rel_err

def _ref(x):
    x = np.asarray(x, dtype=np.float64)
    min_val = x.min()
    max_val = x.max()
    if max_val == min_val:
        s = 1.0
    else:
        s = (max_val - min_val) / 255.0
    z_float = -min_val / s
    zp = int(round(z_float))
    zp = np.clip(zp, -128, 127)
    q = np.round(x / s + zp).astype(np.int32)
    q = np.clip(q, -128, 127)
    deq = (q.astype(np.float64) - zp) * s
    return deq, int(zp)

def grade(sol, fx):
    cases = [
        np.array([0.0, 1.0, -2.5]),
        np.random.randn(10),
        np.linspace(-100, 100, 50),
        np.full((3,4), 42.0),
        np.random.rand(20)*200-50,
    ]
    ok_rel = 1.0
    ok_zp = 1.0
    for x in cases:
        try:
            deq, zp = sol.asymmetric_quant_round_trip(x)
        except Exception:
            return {"rel_err": 0.0, "zero_point_match": 0.0}
        ref_deq, ref_zp = _ref(x)
        rel = rel_err(x, deq)
        if rel > 0.01:
            ok_rel = 0.0
        if zp != ref_zp:
            ok_zp = 0.0
    return {"rel_err": ok_rel, "zero_point_match": ok_zp}
