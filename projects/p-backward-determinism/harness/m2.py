import numpy as np

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from det.rng import fix_state
    m = {"seed_fixed": 0.0}
    try:
        s1 = fix_state(42)
        s2 = fix_state(42)
        if np.allclose(s1, s2):
            m["seed_fixed"] = 1.0
    except Exception:
        pass
    return m
