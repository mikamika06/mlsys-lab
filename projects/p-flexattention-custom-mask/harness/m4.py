import numpy as np

def check(workdir):
    m = {"variable_lengths_ok": 0.0}
    try:
        from mask_engine.predicate import make_mask_predicate
        from mask_engine.compile import compile_attention_mask
        doc_ids = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        pred = make_mask_predicate(doc_ids, window_size=1)
        mask = compile_attention_mask(pred, len(doc_ids))
        if mask.shape == (len(doc_ids), len(doc_ids)):
            m["variable_lengths_ok"] = 1.0
    except Exception:
        pass
    return m
