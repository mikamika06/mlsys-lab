import numpy as np

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from det.trainer import train_run
    m = {"weights_identical": 0.0}
    try:
        w1 = train_run(seed=100, deterministic=True)
        w2 = train_run(seed=100, deterministic=True)
        if np.allclose(w1, w2):
            m["weights_identical"] = 1.0
    except Exception:
        pass
    return m
