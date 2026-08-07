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
    from sampler.chain import apply_top_p, apply_min_p, full_chain, compare_survival

    out = {"topp_err": 1e9, "minp_err": 1e9, "chain_err": 1e9, "survival_match": 0.0}

    logits = np.array([0.1, 2.3, -1.0, 5.0, 4.0, 3.0, 1.0])

    try:
        ref_topp = ref.apply_top_p(logits.copy(), 0.9)
        got_topp = apply_top_p(logits.copy(), 0.9)
        out["topp_err"] = calc_err(ref_topp, got_topp)
    except Exception:
        pass

    try:
        ref_minp = ref.apply_min_p(logits.copy(), 0.1)
        got_minp = apply_min_p(logits.copy(), 0.1)
        out["minp_err"] = calc_err(ref_minp, got_minp)
    except Exception:
        pass

    try:
        ref_chain = ref.full_chain(logits.copy(), [1, 2, 4], 1.2, 2, 5, 0.9, 0.05, 0.8)
        got_chain = full_chain(logits.copy(), [1, 2, 4], 1.2, 2, 5, 0.9, 0.05, 0.8)
        out["chain_err"] = calc_err(ref_chain, got_chain)
    except Exception:
        pass

    try:
        ref_surv = ref.compare_survival(logits.copy(), 0.9, 0.05)
        got_surv = compare_survival(logits.copy(), 0.9, 0.05)
        if ref_surv == got_surv:
            out["survival_match"] = 1.0
    except Exception:
        pass

    return out
