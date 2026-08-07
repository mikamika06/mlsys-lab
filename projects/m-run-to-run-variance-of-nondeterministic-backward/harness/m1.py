import ref


def check(workdir):
    from flashvar.variance import compute_variance
    out = {"variance_measured": 0.0, "rel_err_bounded": 0.0}
    try:
        res = compute_variance(ref.SAMPLE_RUNS)
        if isinstance(res, dict) and "variance" in res and "max_rel_err" in res:
            out["variance_measured"] = 1.0
            ref_res = ref.compute_variance(ref.SAMPLE_RUNS)
            if abs(res["max_rel_err"] - ref_res["max_rel_err"]) < 1e-5:
                out["rel_err_bounded"] = 1.0
    except Exception as e:
        out["_note"] = f"m1 failed: {type(e).__name__}: {str(e)[:120]}"
    return out
