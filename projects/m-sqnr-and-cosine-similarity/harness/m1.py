import numpy as np
import ref


def check(workdir):
    from numval.metrics import cosine_similarity, sqnr

    out = {"sqnr_matches": 0.0, "cos_sim_matches": 0.0}
    sqnr_ok = 0
    cos_ok = 0
    total = len(ref.METRIC_TEST_CASES)

    for i, (y_ref, y_test) in enumerate(ref.METRIC_TEST_CASES):
        want_sqnr = ref.compute_sqnr(y_ref, y_test)
        got_sqnr = sqnr(y_ref, y_test)
        if np.isclose(want_sqnr, got_sqnr, rtol=1e-4, atol=1e-4):
            sqnr_ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i} sqnr mismatch: got {got_sqnr}, ref {want_sqnr}"

        want_cos = ref.compute_cosine_similarity(y_ref, y_test)
        got_cos = cosine_similarity(y_ref, y_test)
        if np.isclose(want_cos, got_cos, rtol=1e-4, atol=1e-4):
            cos_ok += 1
        elif "_note" not in out and sqnr_ok == (i + 1):
            out["_note"] = f"case {i} cos_sim mismatch: got {got_cos}, ref {want_cos}"

    out["sqnr_matches"] = float(sqnr_ok) / total
    out["cos_sim_matches"] = float(cos_ok) / total
    return out
