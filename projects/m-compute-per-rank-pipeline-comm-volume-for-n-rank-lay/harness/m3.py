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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_broken_sharding": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on valid code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import mlxdist.sharding as sh
    good_sharding = sh.derive_load_balanced_sharding

    def broken_sharding(num_layers, layer_weights, num_ranks=4):
        return [i % num_ranks for i in range(num_layers)]

    sh.derive_load_balanced_sharding = broken_sharding
    import mlxdist
    mlxdist.sharding.derive_load_balanced_sharding = broken_sharding

    try:
        out["catches_broken_sharding"] = 0.0 if _survives(path) else 1.0
    finally:
        sh.derive_load_balanced_sharding = good_sharding
        mlxdist.sharding.derive_load_balanced_sharding = good_sharding

    return out
