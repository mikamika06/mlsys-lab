import tempfile


def check(workdir):
    from cachefootprint.invalidation import check_path_invalidation

    out = {"invalidation_detected": 0.0}
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        try:
            res = check_path_invalidation(tmp1, tmp2)
            if isinstance(res, dict) and res.get("invalidated") is True:
                out["invalidation_detected"] = 1.0
            else:
                out["_note"] = f"unexpected invalidation result: {res}"
        except Exception as e:
            out["_note"] = f"check_path_invalidation failed: {type(e).__name__}: {str(e)[:120]}"
    return out
