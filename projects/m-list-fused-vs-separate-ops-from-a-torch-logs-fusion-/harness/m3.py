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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_illegal_fusion": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on valid reference code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import inductorsched.fusion as fmod
    orig_fuse = fmod.greedy_pointwise_fuse

    def bad_fuse(nodes):
        return [[n['id'] for n in nodes]]

    fmod.greedy_pointwise_fuse = bad_fuse
    import inductorsched
    inductorsched.fusion.greedy_pointwise_fuse = bad_fuse

    try:
        catches = not _survives(path)
        out["catches_illegal_fusion"] = 1.0 if catches else 0.0
    finally:
        fmod.greedy_pointwise_fuse = orig_fuse
        inductorsched.fusion.greedy_pointwise_fuse = orig_fuse

    return out
