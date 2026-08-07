import ref


def check(workdir):
    try:
        from zero_estimator.memory import calculate_prefetch_depth
    except ImportError:
        return {"_note": "failed to import calculate_prefetch_depth"}

    out = {"gather_rel_err": 1.0, "depths_exact": 0.0}
    depths_ok = 0
    max_err = 0.0

    for i, cfg in enumerate(ref.CONFIGS):
        lp = cfg["layer_params"]
        ct = cfg["compute_times"]
        bw = cfg["bandwidth"]

        want_gt, want_d = ref.calculate_prefetch_depth(lp, ct, bw)
        try:
            got_gt, got_d = calculate_prefetch_depth(lp, ct, bw)

            if got_d == want_d:
                depths_ok += 1
            elif "_note" not in out:
                out["_note"] = f"cfg {i}: depths want {want_d}, got {got_d}"

            if len(got_gt) == len(want_gt):
                for g, w in zip(got_gt, want_gt):
                    if w != 0:
                        max_err = max(max_err, abs(g - w) / w)
                    else:
                        max_err = max(max_err, abs(g - w))
            else:
                max_err = 1.0
        except Exception as e:
            if "_note" not in out: out["_note"] = f"depth error: {e}"
            max_err = 1.0

    out["depths_exact"] = float(depths_ok) / len(ref.CONFIGS)
    out["gather_rel_err"] = float(max_err)
    return out
