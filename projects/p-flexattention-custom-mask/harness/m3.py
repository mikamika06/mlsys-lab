import numpy as np
import ref

def check(workdir):
    m = {"memory_savings_ratio": 0.0}
    try:
        from mask_engine.predicate import make_mask_predicate
        from mask_engine.block import build_block_mask
        doc_ids, window_size, block_size, seq_len = ref.get_reference_data()
        pred = make_mask_predicate(doc_ids, window_size)
        bm = build_block_mask(seq_len, block_size, pred)

        total_blocks = bm.size
        active_blocks = np.sum(bm)
        ratio = total_blocks / float(active_blocks) if active_blocks > 0 else 1.0
        m["memory_savings_ratio"] = float(ratio)
    except Exception:
        pass
    return m
