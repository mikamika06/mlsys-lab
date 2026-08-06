import ref
import numpy as np

def check(workdir):
    from kd_sweep.gradient import verify_gradient_scaling
    tl = np.array([[2.0, 1.0, 0.5]])
    sl = np.array([[1.5, 1.2, 0.3]])
    T = 2.5
    want = ref.verify_gradient_scaling(tl, sl, T)
    try:
        got = verify_gradient_scaling(tl, sl, T)
    except Exception as e:
        return {"grad_rel_err": 1.0, "_note": str(e)}

    rel_err = float(np.abs(want - got) / (np.abs(want) + 1e-8))
    return {"grad_rel_err": rel_err}
