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

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_exclude": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        return out

    try:
        first = _run(path)
    except Exception:
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import quant_recipes.allocator as alloc

    good_opt = alloc.optimal_alloc

    def bad_opt(profile, exclude, budget, base_bits=16):
        return good_opt(profile, [], budget, base_bits)

    alloc.optimal_alloc = bad_opt
    try:
        survives = False
        try:
            _run(path)
            survives = True
        except Exception:
            pass
        out["catches_ignored_exclude"] = 0.0 if survives else 1.0
    finally:
        alloc.optimal_alloc = good_opt

    return out
