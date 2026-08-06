import ref
import numpy as np

def check(workdir):
    from schema_opt.mask import compute_token_mask
    out = {"mask_match": 0.0}
    allowed = [10, 20, 30, 40]
    want = ref.compute_token_mask(ref.VOCAB_SIZE, allowed)
    got = compute_token_mask(ref.VOCAB_SIZE, allowed)
    if np.array_equal(want, got):
        out["mask_match"] = 1.0
    return out
