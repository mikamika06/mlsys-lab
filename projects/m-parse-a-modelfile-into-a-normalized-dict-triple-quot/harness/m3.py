import importlib.util
import os
import sys

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.dirname(path)))
    spec.loader.exec_module(mod)
    sys.path.pop(0)
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_overwritten_stops": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    sys.path.insert(0, workdir)
    import modelfile.parser as p
    good_parse = p.parse

    def bad_parse(text):
        ast = good_parse(text)
        for k in ast.get("PARAMETER", {}):
            if len(ast["PARAMETER"][k]) > 1:
                ast["PARAMETER"][k] = [ast["PARAMETER"][k][-1]]
        return ast

    p.parse = bad_parse
    try:
        if not _survives(path):
            out["catches_overwritten_stops"] = 1.0
        else:
            out["_note"] = "tests passed despite parser dropping previous stop tokens"
    finally:
        p.parse = good_parse
        sys.path.pop(0)

    return out
