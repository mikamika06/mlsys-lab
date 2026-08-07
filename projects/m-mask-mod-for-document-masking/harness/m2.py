import ref
import numpy as np

def check(workdir):
    from docmask.masks import prefix_lm_mask_mod
    from docmask.sparsity import block_sparsity_fraction

    out = {"prefix_matched": 0.0, "sparsity_matched": 0.0}

    prefix_ok = 0
    for p_len in ref.TEST_CASES_PREFIX:
        ref_mask = ref.ref_prefix_lm_mask_mod(p_len)
        try:
            learner_mask = prefix_lm_mask_mod(p_len)
        except Exception:
            break

        matched = True
        seq_len = 8
        for q in range(seq_len):
            for kv in range(seq_len):
                if bool(ref_mask(0, 0, q, kv)) != bool(learner_mask(0, 0, q, kv)):
                    matched = False
                    break
            if not matched:
                break
        if matched:
            prefix_ok += 1

    if prefix_ok == len(ref.TEST_CASES_PREFIX):
        out["prefix_matched"] = 1.0

    try:
        def sample_mask(b, h, q, kv):
            return q >= kv
        ref_frac = ref.ref_block_sparsity_fraction(sample_mask, 256, 128)
        learner_frac = block_sparsity_fraction(sample_mask, 256, 128)
        if abs(ref_frac - learner_frac) < 1e-5:
            out["sparsity_matched"] = 1.0
    except Exception:
        pass

    return out
