import ref
import numpy as np
import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from moe_routing.router import assign_tokens, compute_load_imbalance

    out = {"assignments_match": 0.0, "imbalance_match": 0.0}
    try:
        for temp in [0.5, 1.0, 2.0]:
            got_assign = assign_tokens(ref.LOGITS, temp, 0.15)
            want_assign = ref.assign_tokens(ref.LOGITS, temp, 0.15)
            if not np.array_equal(got_assign, want_assign):
                out["_note"] = f"assignments mismatch at temp {temp}"
                return out

            got_imb = compute_load_imbalance(want_assign, 8)
            want_imb = ref.compute_load_imbalance(want_assign, 8)
            if abs(got_imb - want_imb) > 1e-5:
                out["_note"] = f"imbalance mismatch at temp {temp}"
                return out

        out["assignments_match"] = 1.0
        out["imbalance_match"] = 1.0
    except Exception as e:
        out["_note"] = f"Error: {e}"
    return out
