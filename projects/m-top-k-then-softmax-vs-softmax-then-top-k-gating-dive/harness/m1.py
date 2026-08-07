import numpy as np
import ref


def check(workdir):
    from moegating.gating import (
        analyze_gating_divergence,
        softmax_then_top_k,
        top_k_then_softmax,
    )

    out = {"max_abs_err": 0.0, "gating_matches": 0.0}
    errs = []

    for logits in ref.LOGITS_DATA:
        want_w1, want_idx1 = ref.top_k_then_softmax(logits, ref.TOP_K)
        got_w1, got_idx1 = top_k_then_softmax(logits, ref.TOP_K)

        want_w2, want_idx2 = ref.softmax_then_top_k(logits, ref.TOP_K)
        got_w2, got_idx2 = softmax_then_top_k(logits, ref.TOP_K)

        want_div = ref.analyze_gating_divergence(logits, ref.TOP_K)
        got_div = analyze_gating_divergence(logits, ref.TOP_K)

        errs.append(np.max(np.abs(want_w1 - got_w1)))
        errs.append(np.max(np.abs(want_w2 - got_w2)))
        errs.append(
            abs(float(want_div["max_abs_diff"]) - float(got_div["max_abs_diff"]))
        )
        errs.append(
            np.max(
                np.abs(
                    np.asarray(want_div["cosine_sim"])
                    - np.asarray(got_div["cosine_sim"])
                )
            )
        )

        if (
            not np.array_equal(want_idx1, got_idx1)
            or not np.array_equal(want_idx2, got_idx2)
        ):
            out["_note"] = "top-k indices mismatched"
            out["max_abs_err"] = 1.0
            return out

    max_err = float(max(errs)) if errs else 0.0
    out["max_abs_err"] = max_err
    if max_err <= 1e-5:
        out["gating_matches"] = 1.0
    return out
