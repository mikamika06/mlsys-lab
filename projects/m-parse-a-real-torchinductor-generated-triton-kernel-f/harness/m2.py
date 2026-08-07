import ref


def check(workdir):
    from inductor_parse.diff import diff_configs
    from inductor_parse.autotune import find_argmin_config

    out = {"diffs_matched": 0.0, "argmin_matched": 0.0}

    cfg1 = {"XBLOCK": 32, "RBLOCK": 128, "num_warps": 4, "num_stages": 2}
    cfg2 = {"XBLOCK": 128, "RBLOCK": 128, "num_warps": 8, "num_stages": 4}

    want_diff = ref.diff_configs(cfg1, cfg2)
    try:
        got_diff = diff_configs(cfg1, cfg2)
        if got_diff == want_diff:
            out["diffs_matched"] = 1.0
        else:
            out["_note"] = f"diff mismatch: got {got_diff}, want {want_diff}"
    except Exception as e:
        out["_note"] = f"diff_configs raised {type(e).__name__}: {e}"
        return out

    want_argmin = ref.find_argmin_config(ref.CANDIDATE_LOGS)
    try:
        got_argmin = find_argmin_config(ref.CANDIDATE_LOGS)
        if got_argmin == want_argmin:
            out["argmin_matched"] = 1.0
        elif "_note" not in out:
            out["_note"] = f"argmin mismatch: got {got_argmin}, want {want_argmin}"
    except Exception as e:
        if "_note" not in out:
            out["_note"] = f"find_argmin_config raised {type(e).__name__}: {e}"

    return out
