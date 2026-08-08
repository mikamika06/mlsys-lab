import importlib.util
import os
import ref

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_inclusive_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import trace_parser.proton as p
    good = p.analyze_proton

    def inclusive_proton(events):
        if not events:
            return {}
        inc_time = {}
        enters = {}
        for ev in events:
            if ev["type"] == "enter":
                enters[ev["region"]] = ev["time"]
            else:
                inc_time[ev["region"]] = inc_time.get(ev["region"], 0.0) + (ev["time"] - enters[ev["region"]])
        total = max(e["time"] for e in events) - min(e["time"] for e in events)
        if total == 0:
            return {}
        return {k: (v / total) * 100.0 for k, v in inc_time.items()}

    p.analyze_proton = inclusive_proton
    try:
        if not _survives(path):
            out["catches_inclusive_bug"] = 1.0
    finally:
        p.analyze_proton = good

    return out
