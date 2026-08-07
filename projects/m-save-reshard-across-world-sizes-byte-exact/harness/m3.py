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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_reshard": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import dcp.reshard as r_mod
    good_reshard = r_mod.reshard_state_dict

    def bad_reshard(source_dicts, s_ws, t_ws):
        res = good_reshard(source_dicts, s_ws, t_ws)
        for d in res:
            if "chunk" in d:
                d["chunk"] = d["chunk"] * 0
        return res

    r_mod.reshard_state_dict = bad_reshard
    import dcp
    dcp.reshard_state_dict = bad_reshard
    try:
        survives = _survives(path)
        out["catches_bad_reshard"] = 0.0 if survives else 1.0
    finally:
        r_mod.reshard_state_dict = good_reshard
        dcp.reshard_state_dict = good_reshard
    return out
