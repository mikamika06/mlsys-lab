import ref
import numpy as np

def check(workdir):
    from docmask.masks import document_mask_mod
    out = {"doc_masks_matched": 0.0}
    ok = 0
    total = len(ref.TEST_CASES_DOC)
    for doc_ids in ref.TEST_CASES_DOC:
        ref_mask = ref.ref_document_mask_mod(doc_ids)
        try:
            learner_mask = document_mask_mod(doc_ids)
        except Exception:
            break

        matched = True
        seq_len = doc_ids.shape[1]
        for q in range(seq_len):
            for kv in range(seq_len):
                if bool(ref_mask(0, 0, q, kv)) != bool(learner_mask(0, 0, q, kv)):
                    matched = False
                    break
            if not matched:
                break
        if matched:
            ok += 1
    if ok == total:
        out["doc_masks_matched"] = 1.0
    return out
