import ref
import numpy as np

def check(workdir):
    from gptq.core import act_order_perm, find_damping
    
    out = {"perm_match": 0.0, "damp_match": 0.0, "argmin_index": 0.0}
    perm_ok = 0
    damp_ok = 0
    argmin_ok = 0
    
    for H, _, _ in ref.FIXTURES:
        try:
            got_p = act_order_perm(H)
            want_p = ref.act_order_perm(H)
            if np.array_equal(got_p, want_p):
                perm_ok += 1
            if len(got_p) > 0 and len(want_p) > 0 and got_p[-1] == want_p[-1]:
                argmin_ok += 1
        except Exception:
            pass
            
        try:
            got_d = find_damping(H)
            want_d = ref.find_damping(H)
            if abs(got_d - want_d) < 1e-9:
                damp_ok += 1
        except Exception:
            pass

    out["perm_match"] = perm_ok / len(ref.FIXTURES)
    out["argmin_index"] = argmin_ok / len(ref.FIXTURES)
    out["damp_match"] = damp_ok / len(ref.FIXTURES)
    return out
