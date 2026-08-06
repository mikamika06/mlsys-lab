import ref
import numpy as np

def check(workdir):
    from gptq.core import lazy_batch_update
    
    out = {"block_update_match": 0.0, "preserves_output": 0.0}
    ok = 0
    
    for H, W, errors in ref.FIXTURES:
        H_inv = np.linalg.inv(H + np.eye(H.shape[0]))
        W_got = W.copy()
        W_want = W.copy()
        
        try:
            W_got = lazy_batch_update(W_got, H_inv, errors, 16, 16)
            W_want = ref.lazy_batch_update(W_want, H_inv, errors, 16, 16)
            if np.allclose(W_got, W_want):
                ok += 1
        except Exception:
            pass

    out["block_update_match"] = ok / len(ref.FIXTURES)
    out["preserves_output"] = ok / len(ref.FIXTURES)
    return out
