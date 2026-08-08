import numpy as np
import ref

def check(workdir):
    out = {"ladder_keys": 0.0, "ladder_bytes": 0.0, "ladder_errs": 0.0}
    try:
        from quant.ladder import evaluate_ladder
    except ImportError:
        out["_note"] = "Failed to import quant.ladder"
        return out

    w = ref.FIXTURE_W[:64, :64]
    try:
        got = evaluate_ladder(w, group_size=32)
        want = ref.evaluate_ladder(w, group_size=32)
    except Exception as e:
        out["_note"] = f"ladder execution failed: {type(e).__name__}"
        return out

    if not got or not isinstance(got, list) or len(got) != 4:
        out["_note"] = "ladder did not return a list of 4 items"
        return out

    out["ladder_keys"] = 1.0
    ok_b = 0
    ok_e = 0

    for g_dict, w_dict in zip(got, want):
        if g_dict.get("meta_bytes") == w_dict["meta_bytes"]:
            ok_b += 1
        if abs(g_dict.get("max_abs_err", 0) - w_dict["max_abs_err"]) < 1e-4:
            ok_e += 1

    if ok_b == 4:
        out["ladder_bytes"] = 1.0
    else:
        out["_note"] = "metadata bytes logic is incorrect"

    if ok_e == 4:
        out["ladder_errs"] = 1.0

    return out
