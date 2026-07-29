import importlib.util
import os


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_merged_phases": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import coldpath.phases as ph
    good = ph.build_phases

    def merged(events):
        marks = {}
        for e in events:
            marks[e["name"]] = e["t"]
        daemon_pair = "daemon_start" in marks and "daemon_end" in marks
        load_pair = "load_start" in marks and "load_end" in marks
        out_ = []
        if daemon_pair and load_pair:
            out_.append({"name": "daemon", "start": marks["daemon_start"],
                         "end": marks["load_end"],
                         "duration_ms": marks["load_end"] - marks["daemon_start"]})
        elif daemon_pair:
            out_.append({"name": "daemon", "start": marks["daemon_start"],
                         "end": marks["daemon_end"],
                         "duration_ms": marks["daemon_end"] - marks["daemon_start"]})
        elif load_pair:
            out_.append({"name": "load", "start": marks["load_start"],
                         "end": marks["load_end"],
                         "duration_ms": marks["load_end"] - marks["load_start"]})
        for name in ("prefill", "decode"):
            sk, ek = name + "_start", name + "_end"
            if sk in marks and ek in marks:
                out_.append({"name": name, "start": marks[sk], "end": marks[ek],
                             "duration_ms": marks[ek] - marks[sk]})
        return out_

    ph.build_phases = merged
    import coldpath
    coldpath.build_phases = merged
    try:
        out["catches_merged_phases"] = 0.0 if _survives(path) else 1.0
    finally:
        ph.build_phases = good
        coldpath.build_phases = good
    return out
