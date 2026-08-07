import ref
import numpy as np

def check(workdir):
    from app.decoder import apply_grammar_mask
    logits = np.array([1.0, 2.0, -1.0, 4.0])
    allowed = [1, 3]
    res = apply_grammar_mask(logits, allowed)
    m = {"mask_ok": 0.0}
    if res[0] == -np.inf and res[2] == -np.inf and res[1] == 2.0 and res[3] == 4.0:
        m["mask_ok"] = 1.0
    return m
