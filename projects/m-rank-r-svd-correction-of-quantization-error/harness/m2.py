import ref
import numpy as np


def check(workdir):
    from quantcorr.strategies import (
        simulate_rtn,
        simulate_gptq,
        simulate_rotation_gptq,
        simulate_autoround,
    )

    out = {"strategies_matched": 0.0}
    ok = 0
    w = ref.TEST_WEIGHTS[0]
    hinv = ref.HINVS[0]
    rot = ref.ROT_MATRICES[0]

    try:
        res_rtn = simulate_rtn(w, 4)
        if isinstance(res_rtn, np.ndarray) and res_rtn.shape == w.shape:
            ok += 1
        res_gptq = simulate_gptq(w, hinv, 4)
        if isinstance(res_gptq, np.ndarray) and res_gptq.shape == w.shape:
            ok += 1
        res_rot = simulate_rotation_gptq(w, hinv, 4, rot)
        if isinstance(res_rot, np.ndarray) and res_rot.shape == w.shape:
            ok += 1
        res_ar = simulate_autoround(w, 4)
        if isinstance(res_ar, np.ndarray) and res_ar.shape == w.shape:
            ok += 1
    except Exception as e:
        out["_note"] = f"Strategy execution error: {type(e).__name__}: {str(e)[:120]}"

    out["strategies_matched"] = float(ok)
    return out
