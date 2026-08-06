import ref
import tempfile

def check(workdir):
    from compcache.invalidation import check_cache_behavior
    out = {"invalidation_behavior_correct": 0.0}
    with tempfile.TemporaryDirectory() as d:
        try:
            res = check_cache_behavior(d)
            if isinstance(res, dict) and res.get("identical_hit") and res.get("invalidated"):
                out["invalidation_behavior_correct"] = 1.0
            else:
                out["_note"] = f"result incorrect: {res}"
        except Exception as e:
            out["_note"] = f"error: {type(e).__name__}: {str(e)[:120]}"
    return out
