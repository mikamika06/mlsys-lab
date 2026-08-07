import ref


def check(workdir):
    from interop.diffing import diff_model_weights

    out = {"diffs_matched": 0.0, "max_diff_below_tol": 0.0}
    want = ref.diff_model_weights(ref.GGUF_WEIGHTS, ref.CONVERT_WEIGHTS)
    got = diff_model_weights(ref.GGUF_WEIGHTS, ref.CONVERT_WEIGHTS)

    if (
        isinstance(got, dict)
        and got.get("common_keys") == want["common_keys"]
        and got.get("matched_keys") == want["matched_keys"]
    ):
        out["diffs_matched"] = 1.0

    if isinstance(got, dict) and abs(got.get("max_abs_diff", 1.0) - want["max_abs_diff"]) < 1e-6:
        out["max_diff_below_tol"] = 1.0
    else:
        out["_note"] = f"Got max_abs_diff {got.get('max_abs_diff')}, expected {want['max_abs_diff']}"

    return out
