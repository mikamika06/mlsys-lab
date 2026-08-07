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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0,
           "catches_dominated": 0.0, "catches_missing": 0.0}

    if not os.path.isfile(path):
        return out

    import compress.api as api

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_pareto = api.pareto_frontier

    def bad_pareto1(pts):
        return sorted(list(set(pts)), key=lambda x: x[0])

    api.pareto_frontier = bad_pareto1
    try:
        out["catches_dominated"] = 0.0 if _survives(path) else 1.0
    finally:
        api.pareto_frontier = good_pareto

    def bad_pareto2(pts):
        opt = good_pareto(pts)
        return opt[1:] if len(opt) > 1 else []

    api.pareto_frontier = bad_pareto2
    try:
        out["catches_missing"] = 0.0 if _survives(path) else 1.0
    finally:
        api.pareto_frontier = good_pareto

    return out
