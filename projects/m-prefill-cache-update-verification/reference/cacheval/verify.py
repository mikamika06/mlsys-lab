import numpy as np


def verify_prefill_update(reference_cache, candidate_cache, max_abs_err):
    if len(reference_cache) != len(candidate_cache):
        return False
    for ref_layer, cand_layer in zip(reference_cache, candidate_cache):
        ref_k, ref_v = ref_layer
        cand_k, cand_v = cand_layer
        if ref_k.shape != cand_k.shape or ref_v.shape != cand_v.shape:
            return False
        if not np.all(np.abs(ref_k - cand_k) <= max_abs_err):
            return False
        if not np.all(np.abs(ref_v - cand_v) <= max_abs_err):
            return False
    return True
