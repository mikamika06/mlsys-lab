import ref
import numpy as np


def check(workdir):
    from blockinf.metric import compute_block_influence
    model = ref.get_test_fixture()
    want = ref.compute_block_influence(model)
    try:
        got = compute_block_influence(model)
    except Exception as e:
        return {"max_rel_err": 1.0, "_note": f"raised {type(e).__name__}: {e}"}

    want_arr = np.array(want, dtype=float)
    got_arr = np.array(got, dtype=float)
    if want_arr.shape != got_arr.shape:
        return {"max_rel_err": 1.0, "_note": f"shape mismatch: got {got_arr.shape}, want {want_arr.shape}"}

    rel_err = np.max(np.abs(want_arr - got_arr) / (np.abs(want_arr) + 1e-8))
    return {"max_rel_err": float(rel_err)}
