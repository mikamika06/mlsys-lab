import numpy as np

def _ref(lat, thr, slo):
    valid = np.where(lat <= slo)[0]
    if len(valid) == 0:
        return -1
    max_idx = valid[np.argmax(thr[valid])]
    return int(max_idx)

def grade(sol, fx) -> dict:
    cases = [
        # simple case with a clear knee
        (np.array([10,12,15,20]), np.array([100,120,110,90]), 18.0),
        # no batch satisfies the SLO
        (np.array([25,30,35]), np.array([80,70,60]), 20.0),
        # multiple batches with equal max throughput before SLO
        (np.array([5,10,15,20]), np.array([100,150,150,140]), 18.0),
        # larger random case
        (np.random.default_rng(42).uniform(5,30,size=50), 
         np.random.default_rng(43).uniform(80,200,size=50), 22.0),
        # edge: last batch is the knee
        (np.array([10,15,20]), np.array([90,110,130]), 25.0)
    ]
    ok = 1.0
    for lat, thr, slo in cases:
        try:
            got = sol.find_batch_size_knee(lat, thr, slo)
        except Exception:
            return {"exact_match": 0.0}
        ref = _ref(lat, thr, slo)
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
