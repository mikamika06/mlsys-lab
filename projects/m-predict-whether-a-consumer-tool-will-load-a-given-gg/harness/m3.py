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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bypassed_compat": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import gguf_interop.compat as c
    good_fn = c.check_tool_compatibility

    def broken_check_tool_compatibility(metadata, tool_profile):
        return {"compatible": True, "reasons": []}

    c.check_tool_compatibility = broken_check_tool_compatibility
    import gguf_interop
    gguf_interop.compat.check_tool_compatibility = broken_check_tool_compatibility

    try:
        out["catches_bypassed_compat"] = 0.0 if _survives(path) else 1.0
    finally:
        c.check_tool_compatibility = good_fn
        gguf_interop.compat.check_tool_compatibility = good_fn

    return out
