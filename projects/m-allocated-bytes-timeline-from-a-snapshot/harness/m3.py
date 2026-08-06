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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_leaks": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on valid reference: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import snaptool.frames as frames_mod
    good_find = frames_mod.find_retaining_frame

    def broken_find_retaining_frame(snapshot):
        return ("dummy.py:0:dummy", 0)

    frames_mod.find_retaining_frame = broken_find_retaining_frame
    import snaptool
    snaptool.frames.find_retaining_frame = broken_find_retaining_frame

    try:
        if _survives(path):
            out["catches_ignored_leaks"] = 0.0
            out["_note"] = "test suite failed to catch broken retaining frame analysis"
        else:
            out["catches_ignored_leaks"] = 1.0
    finally:
        frames_mod.find_retaining_frame = good_find
        snaptool.frames.find_retaining_frame = good_find

    return out
