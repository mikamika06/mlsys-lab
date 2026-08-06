import ref
import numpy as np

def check(workdir):
    from kd_sweep.sweep import run_sweep
    t_list = [1.0, 2.0, 4.0]
    a_list = [0.1, 0.5, 0.9]
    tl = np.array([[2.0, 1.0, 0.5]])
    sl = np.array([[1.5, 1.2, 0.3]])
    lbl = np.array([[1.0, 0.0, 0.0]])

    want = ref.run_sweep(tl, sl, lbl, t_list, a_list)
    try:
        got = run_sweep(tl, sl, lbl, t_list, a_list)
    except Exception as e:
        return {"sweep_rel_err": 1.0, "_note": str(e)}

    rel_err = float(np.max(np.abs(want - got) / (np.abs(want) + 1e-8)))
    return {"sweep_rel_err": rel_err}
