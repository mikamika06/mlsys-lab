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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_mode": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"test failed on good code: {type(e).__name__}: {str(e)[:100]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import quant.modes as qm
    good_fn = qm.analyze_op_compatibility

    def broken_analyze(model_spec, quant_mode="int8"):
        res = good_fn(model_spec, quant_mode)
        for item in res:
            item["executes_float"] = False
        return res

    qm.analyze_op_compatibility = broken_analyze
    import quant
    quant.modes.analyze_op_compatibility = broken_analyze

    try:
        out["catches_invalid_mode"] = 0.0 if _survives(path) else 1.0
    finally:
        qm.analyze_op_compatibility = good_fn
        quant.modes.analyze_op_compatibility = good_fn

    return out
