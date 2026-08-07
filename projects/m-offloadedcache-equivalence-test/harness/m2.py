import sys
sys.path.insert(0, ".")
import ref
import numpy as np

def check(workdir):
    out = {"update_matches": 0.0, "seq_len_matches": 0.0}
    try:
        from kvcache.offloaded import DynamicCache, OffloadedCache
        dc = DynamicCache()
        oc = OffloadedCache()

        k1, v1 = ref.generate_update_states(seq_len=3)
        k2, v2 = ref.generate_update_states(seq_len=2)

        dc.update(k1, v1, 0)
        oc.update(k1, v1, 0)

        dk2, dv2 = dc.update(k2, v2, 0)
        ok2, ov2 = oc.update(k2, v2, 0)

        if np.allclose(dk2, ok2) and np.allclose(dv2, ov2):
            out["update_matches"] = 1.0

        if dc.get_seq_length(0) == oc.get_seq_length(0) and oc.get_seq_length(0) == 5:
            out["seq_len_matches"] = 1.0
    except Exception as e:
        out["_note"] = f"m2 failed with {type(e).__name__}: {str(e)}"
    return out
