import importlib.util
import os
import sys


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


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_overpruning": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        return out

    sys.path.insert(0, workdir)
    try:
        import autotune
        import importlib
        importlib.reload(autotune)
    except Exception:
        sys.path.pop(0)
        return out

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        sys.path.pop(0)
        return out

    if first is None:
        sys.path.pop(0)
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_is_dominated = autotune.is_dominated

    def buggy_is_dominated(config, oom_configs, resource_keys):
        for oom in oom_configs:
            if any(config[k] >= oom[k] for k in resource_keys):
                return True
        return False

    autotune.is_dominated = buggy_is_dominated
    try:
        survives = False
        try:
            survives = _run(path) is True
        except Exception:
            survives = False
        out["catches_overpruning"] = 0.0 if survives else 1.0
    finally:
        autotune.is_dominated = good_is_dominated
        sys.path.pop(0)

    return out
