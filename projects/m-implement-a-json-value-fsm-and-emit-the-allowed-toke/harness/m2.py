import ref
import numpy as np

def check(workdir):
    from jsonfsm.mask import compute_mask, verify_equivalence
    out = {"equivalence_match": 0.0}
    logits = np.array([0.5, 2.3, 1.1, 4.0])
    allowed = [1, 3]
    masked = compute_mask(logits, allowed)
    if verify_equivalence(logits, masked, allowed):
        out["equivalence_match"] = 1.0
    return out
