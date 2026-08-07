import ref
import numpy as np


def check(workdir):
    from gptq.obq import compute_obq_update

    W, invH, _ = ref.generate_inputs()
    out = {"update_matched": 0.0}
    col_idx = 5
    w_col = W[:, col_idx]
    invH_col = invH[:, col_idx]
    invH_val = invH[col_idx, col_idx]
    want = ref.reference_obq_update(w_col, invH_col, invH_val)
    got = compute_obq_update(w_col, invH_col, invH_val)
    if got is not None and np.allclose(got, want, atol=1e-7):
        out["update_matched"] = 1.0
    else:
        out["_note"] = f"OBQ update mismatch. Max diff: {np.max(np.abs(got - want)) if got is not None else 'None'}"
    return out
