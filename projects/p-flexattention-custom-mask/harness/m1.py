import numpy as np
import ref

def check(workdir):
    m = {"predicate_ok": 0.0}
    try:
        from mask_engine.predicate import make_mask_predicate
        doc_ids, window_size, _, _ = ref.get_reference_data()
        pred = make_mask_predicate(doc_ids, window_size)
        res = pred(3, 1)
        if isinstance(res, (bool, np.bool_)):
            m["predicate_ok"] = 1.0
    except Exception:
        pass
    return m
