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
           "catches_leak": 0.0, "catches_hr_drop": 0.0}

    if not os.path.isfile(path):
        out["_note"] = "missing"
        return out

    import cache

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"fails on good: {e}"
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    orig_init = cache.PrefixCache.__init__

    def leaky_init(self, bs, alloc, isolation=False, shared_system=False):
        orig_init(self, bs, alloc, isolation=False, shared_system=shared_system)

    cache.PrefixCache.__init__ = leaky_init
    try:
        out["catches_leak"] = 0.0 if _survives(path) else 1.0
    finally:
        cache.PrefixCache.__init__ = orig_init

    def no_share_init(self, bs, alloc, isolation=False, shared_system=False):
        orig_init(self, bs, alloc, isolation=isolation, shared_system=False)

    cache.PrefixCache.__init__ = no_share_init
    try:
        out["catches_hr_drop"] = 0.0 if _survives(path) else 1.0
    finally:
        cache.PrefixCache.__init__ = orig_init

    return out
