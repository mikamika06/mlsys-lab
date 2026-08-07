import importlib.util
import os


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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_missing_shard_bypass": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = (
            f"tests fail on correct reference: {type(e).__name__}: {str(e)[:120]}"
        )
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ggufconv.feasibility as feat

    good = feat.check_feasibility

    def broken(file_list):
        res = good(file_list)
        if res.get("reason") == "missing_shards":
            return {
                "feasible": True,
                "reason": "ok",
                "format": "safetensors",
                "shard_count": 2,
            }
        return res

    feat.check_feasibility = broken
    import ggufconv

    ggufconv.feasibility.check_feasibility = broken
    try:
        out["catches_missing_shard_bypass"] = 0.0 if _survives(path) else 1.0
    finally:
        feat.check_feasibility = good
        ggufconv.feasibility.check_feasibility = good
    return out
