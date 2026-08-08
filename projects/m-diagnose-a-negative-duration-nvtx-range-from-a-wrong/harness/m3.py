import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_invalid_analysis": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import nvtxprof.mac as mac_mod
    import nvtxprof.nvtx as nvtx_mod

    good_nvtx = nvtx_mod.diagnose_nvtx_mismatches
    good_mac = mac_mod.analyze_mac_trace

    def broken_nvtx(events):
        stack = []
        ranges = []
        neg_ranges = []
        unclosed = []
        orphans = []
        for evt in events:
            etype = evt["type"]
            ts = evt["timestamp"]
            if etype == "push":
                stack.append(evt)
            elif etype == "pop":
                if stack:
                    push_evt = stack.pop()
                    dur = ts - push_evt["timestamp"]
                    r = {
                        "name": push_evt["name"],
                        "thread_id": evt["thread_id"],
                        "start": push_evt["timestamp"],
                        "end": ts,
                        "duration": dur,
                    }
                    ranges.append(r)
                    if dur < 0:
                        neg_ranges.append(r)
                else:
                    orphans.append({
                        "thread_id": evt["thread_id"],
                        "timestamp": ts,
                    })
        for push_evt in stack:
            unclosed.append({
                "name": push_evt["name"],
                "thread_id": push_evt["thread_id"],
                "timestamp": push_evt["timestamp"],
            })
        return {
            "ranges": ranges,
            "negative_ranges": neg_ranges,
            "unclosed_pushes": unclosed,
            "orphan_pops": orphans,
        }

    def broken_mac(trace_events, target_phases):
        res = good_mac(trace_events, target_phases)
        for p in target_phases:
            res["phase_metrics"][p]["self_time"] = res["phase_metrics"][p][
                "total_time"
            ]
        res["rankings"] = sorted(
            target_phases,
            key=lambda p: (-res["phase_metrics"][p]["self_time"], p),
        )
        return res

    nvtx_mod.diagnose_nvtx_mismatches = broken_nvtx
    mac_mod.analyze_mac_trace = broken_mac

    try:
        out["catches_invalid_analysis"] = 0.0 if _survives(path) else 1.0
    finally:
        nvtx_mod.diagnose_nvtx_mismatches = good_nvtx
        mac_mod.analyze_mac_trace = good_mac

    return out
