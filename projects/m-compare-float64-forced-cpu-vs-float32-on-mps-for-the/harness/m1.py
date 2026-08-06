import sys


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"backend_checked": 0.0}
    try:
        from mpsbench.device import check_mps_support
        res = check_mps_support()
        if isinstance(res, dict) and "is_built" in res and "is_available" in res:
            if isinstance(res["is_built"], bool) and isinstance(res["is_available"], bool):
                out["backend_checked"] = 1.0
            else:
                out["_note"] = f"Values must be boolean, got {res}"
        else:
            out["_note"] = f"Expected dict with 'is_built' and 'is_available', got {res}"
    except Exception as e:
        out["_note"] = f"check_mps_support failed: {type(e).__name__}: {e}"
    return out
