import importlib.util
import os

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True

def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_count_sorted_stalls": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import warpanalyze.stalls as stalls_mod
    good_func = stalls_mod.top_stall_reasons

    def broken_count_sorted_top_stalls(warp_state_stats, k=3):
        ranked = []
        for entry in warp_state_stats:
            reason = entry["reason"]
            total_cycles = float(entry["total_stall_cycles"])
            total_insts = float(entry["total_executed_instructions"])
            avg_cpi = total_cycles / total_insts if total_insts > 0 else 0.0
            ranked.append((total_cycles, avg_cpi, reason))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [{"reason": r, "avg_cpi": cpi} for _, cpi, r in ranked[:k]]

    stalls_mod.top_stall_reasons = broken_count_sorted_top_stalls
    import warpanalyze
    warpanalyze.stalls.top_stall_reasons = broken_count_sorted_top_stalls

    try:
        out["catches_count_sorted_stalls"] = 0.0 if _survives(path) else 1.0
    finally:
        stalls_mod.top_stall_reasons = good_func
        warpanalyze.stalls.top_stall_reasons = good_func

    return out
