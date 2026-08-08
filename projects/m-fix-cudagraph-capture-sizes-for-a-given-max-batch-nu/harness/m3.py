import importlib.util
import os

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    for fn in fns:
        fn()
    return True

def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_parser": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests failed on reference: {type(e).__name__}: {str(e)}"
        return out
    if first is None:
        out["_note"] = "no test functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import specengine.parser as p
    orig = p.parse_trt_llm_log

    def broken_parse(log):
        res = orig(log)
        res["peak_memory_mb"] = 0.0
        return res

    p.parse_trt_llm_log = broken_parse
    import specengine
    specengine.parser.parse_trt_llm_log = broken_parse
    try:
        out["catches_bad_parser"] = 0.0 if _survives(path) else 1.0
    finally:
        p.parse_trt_llm_log = orig
        specengine.parser.parse_trt_llm_log = orig
    return out
