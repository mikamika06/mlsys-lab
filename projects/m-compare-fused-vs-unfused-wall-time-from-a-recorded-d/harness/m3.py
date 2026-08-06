import importlib.util
import os
import sys


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
    sys.path.insert(0, workdir)
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_ignored_dtypes": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out[
            "_note"
        ] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import bench_analysis.parser as parser_mod

    orig_extract = parser_mod.extract_tensor_bytes

    def broken_extract(shape, dtype_str):
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        return num_elements

    parser_mod.extract_tensor_bytes = broken_extract
    import bench_analysis.report as report_mod

    report_mod.extract_tensor_bytes = broken_extract

    try:
        survived = _survives(path)
        out["catches_ignored_dtypes"] = 0.0 if survived else 1.0
    finally:
        parser_mod.extract_tensor_bytes = orig_extract
        report_mod.extract_tensor_bytes = orig_extract

    return out
