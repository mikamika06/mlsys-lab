import importlib.util
import os
import ref


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
    import sys
    sys.path.insert(0, workdir)

    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_tolerance_violations": 0.0,
    }

    test_path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(test_path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        res = _run(test_path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on reference code: {type(e).__name__}: {str(e)}"
        return out

    if res is None:
        out["_note"] = "No test_ functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import coreml_exporter.converter as conv
    orig_export = conv.export_and_verify

    def faulty_export(model, example_inputs, eval_inputs, save_path):
        mlmodel, _ = orig_export(model, example_inputs, eval_inputs, save_path)
        return mlmodel, 0.5

    conv.export_and_verify = faulty_export
    try:
        out["catches_tolerance_violations"] = 0.0 if _survives(test_path) else 1.0
    finally:
        conv.export_and_verify = orig_export

    return out
