import numpy as np

def check(workdir):
    m = {"max_memory_mb": 0.0, "output_matches": 0.0}
    try:
        from mask_engine.predicate import make_mask_predicate
        from mask_engine.block import build_block_mask
        doc_ids = [0] * 1000
        pred = make_mask_predicate(doc_ids, window_size=16)
        bm = build_block_mask(1000, 64, pred)

        m["max_memory_mb"] = 1.0
        if bm.shape == (16, 16):
            m["output_matches"] = 1.0
    except Exception:
        pass
    return m
