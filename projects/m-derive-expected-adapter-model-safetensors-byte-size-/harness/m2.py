import ref


def check(workdir):
    from peftutils.size import compute_storage_ratio

    out = {"ratios_matched": 0.0}
    try:
        r1 = compute_storage_ratio(200, 1000)
        r2 = compute_storage_ratio(0, 500)
        if abs(r1 - 0.2) < 1e-5 and abs(r2 - 0.0) < 1e-5:
            out["ratios_matched"] = 1.0
        else:
            out["_note"] = f"got ratios {r1}, {r2}, expected 0.2, 0.0"
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {str(e)}"
    return out
