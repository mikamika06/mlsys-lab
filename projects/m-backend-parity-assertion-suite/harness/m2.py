import ref
import numpy as np


def check(workdir):
    from hf_attn.suite import build_repro_case, assert_parity_on_valid
    out = {"exact_match": 0.0}
    
    try:
        q, k, v, mask = build_repro_case(3, 5, [5, 2, 0])
        wq, wk, wv, wmask = ref.build_repro_case(3, 5, [5, 2, 0])
        if not (np.allclose(q, wq) and np.allclose(mask, wmask)):
            return out
    except Exception:
        return out
        
    def r_fn(q, k, v, mask): 
        return np.zeros_like(q)
        
    def t_fn(q, k, v, mask): 
        out = np.zeros_like(q)
        out[1, 2:] = 1.0
        return out
        
    try:
        diff1 = assert_parity_on_valid(q, k, v, mask, r_fn, t_fn)
        if abs(diff1) > 1e-6:
            return out
            
        def t_fn2(q, k, v, mask):
            out = np.zeros_like(q)
            out[1, 1] = 2.5
            return out
            
        diff2 = assert_parity_on_valid(q, k, v, mask, r_fn, t_fn2)
        if abs(diff2 - 2.5) > 1e-6:
            return out
    except Exception:
        return out
        
    out["exact_match"] = 1.0
    return out
