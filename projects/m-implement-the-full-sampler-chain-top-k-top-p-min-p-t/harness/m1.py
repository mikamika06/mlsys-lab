import ref
import numpy as np

def calc_err(ref_arr, got_arr):
    if ref_arr.shape != got_arr.shape:
        return 1e9
    mask = (ref_arr != got_arr)
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(ref_arr[mask] - got_arr[mask])))

def check(workdir):
    from sampler.chain import apply_repetition_penalty, apply_temperature, apply_top_k

    out = {"penalty_err": 1e9, "temp_err": 1e9, "topk_err": 1e9}

    logits = np.array([-2.0, 1.0, 3.0, -1.0, 0.0])
    history = [1, 3]
    penalty = 1.2

    try:
        ref_pen = ref.apply_repetition_penalty(logits.copy(), history, penalty, 1)
        got_pen = apply_repetition_penalty(logits.copy(), history, penalty, 1)
        out["penalty_err"] = calc_err(ref_pen, got_pen)
    except Exception:
        pass

    try:
        ref_temp = ref.apply_temperature(logits.copy(), 0.5)
        got_temp = apply_temperature(logits.copy(), 0.5)
        out["temp_err"] = calc_err(ref_temp, got_temp)
    except Exception:
        pass

    logits2 = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
    try:
        ref_topk = ref.apply_top_k(logits2.copy(), 2)
        got_topk = apply_top_k(logits2.copy(), 2)
        out["topk_err"] = calc_err(ref_topk, got_topk)
    except Exception:
        pass

    return out
