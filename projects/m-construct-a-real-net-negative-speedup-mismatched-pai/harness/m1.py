import sys
import numpy as np

def check(workdir):
    sys.path.insert(0, workdir)
    import spec_fail.metrics as m

    out = {"acceptance_rate_match": 0.0, "speedup_match": 0.0, "collapse_match": 0.0}

    p = np.array([0.1, 0.7, 0.2])
    q = np.array([0.2, 0.5, 0.3])
    rate_want = 0.8
    try:
        rate_got = m.expected_acceptance_rate(p, q)
        if abs(rate_got - rate_want) < 1e-5:
            out["acceptance_rate_match"] = 1.0
    except NotImplementedError:
        pass

    speedup_want = 2.270769230769231
    try:
        speedup_got = m.expected_speedup(p, q, 3, 10.0, 100.0)
        if abs(speedup_got - speedup_want) < 1e-4:
            out["speedup_match"] = 1.0
    except NotImplementedError:
        pass

    p_in, q_in = np.array([0.8, 0.2]), np.array([0.9, 0.1])
    p_out, q_out = np.array([0.2, 0.8]), np.array([0.9, 0.1])
    collapse_want = -0.6
    try:
        collapse_got = m.acceptance_collapse(p_in, q_in, p_out, q_out)
        if abs(collapse_got - collapse_want) < 1e-4:
            out["collapse_match"] = 1.0
    except NotImplementedError:
        pass

    return out
