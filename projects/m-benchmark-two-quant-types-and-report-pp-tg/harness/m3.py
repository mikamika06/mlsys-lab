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


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_swapped_metrics": 0.0}
    
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        res = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if res is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import llambench.parser as parser_mod
    orig_extract = parser_mod.extract_quant_metrics

    def broken_extract(parsed_data, quant_type):
        res = orig_extract(parsed_data, quant_type)
        return {"pp": res["tg"], "tg": res["pp"]}

    parser_mod.extract_quant_metrics = broken_extract
    try:
        failed = False
        try:
            _run(path)
        except Exception:
            failed = True
        out["catches_swapped_metrics"] = 1.0 if failed else 0.0
    finally:
        parser_mod.extract_quant_metrics = orig_extract

    return out
