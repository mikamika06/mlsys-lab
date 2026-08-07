import importlib.util
import os
import sys


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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_ignored_idle_states": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
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

    import profile_parser.parser as p
    good_parse = p.parse_gpu_active_residency

    def broken_parse_ignores_idle(text):
        samples = text.split("*** Sampled system activity")
        results = []
        for sample in samples:
            if not sample.strip():
                continue
            import re
            m = re.search(r"GPU\s+HW\s+active\s+residency:\s*([\d\.]+)%", sample, re.IGNORECASE)
            if m:
                results.append(float(m.group(1)))
            else:
                results.append(0.0)
        return results

    p.parse_gpu_active_residency = broken_parse_ignores_idle
    try:
        out["catches_ignored_idle_states"] = 0.0 if _survives(path) else 1.0
    finally:
        p.parse_gpu_active_residency = good_parse

    return out
