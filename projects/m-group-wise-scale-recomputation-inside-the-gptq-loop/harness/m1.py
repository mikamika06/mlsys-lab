import ref
import numpy as np


def check(workdir):
    from gptq.scales import compute_group_scales

    w = ref.W_TEST
    want = ref.compute_group_scales(w, ref.GROUP_SIZE, ref.BITS)
    try:
        got = compute_group_scales(w, ref.GROUP_SIZE, ref.BITS)
    except Exception as e:
        return {"max_abs_err": float("inf"), "_note": f"raised {type(e).__name__}: {str(e)[:100]}"}

    if got is None or not isinstance(got, np.ndarray) or got.shape != want.shape:
        return {"max_abs_err": float("inf"), "_note": f"invalid output shape or type: {type(got)}"}

    err = float(np.max(np.abs(got - want)))
    out = {"max_abs_err": err}
    if err > 1e-5:
        out["_note"] = f"max abs error {err} exceeds threshold"
    return out
