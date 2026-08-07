import importlib.util
import os


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


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0,
           "catches_no_isolation": 0.0, "catches_broken_system_prefix": 0.0,
           "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    import prefix_cache.cache as pcache
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0
    good_init = pcache.PrefixCache.__init__
    def leaky_init(self, capacity=100, isolate=True):
        good_init(self, capacity=capacity, isolate=False)
    pcache.PrefixCache.__init__ = leaky_init
    try:
        out["catches_no_isolation"] = 0.0 if _survives(path) else 1.0
    finally:
        pcache.PrefixCache.__init__ = good_init
    good_lookup = pcache.PrefixCache.lookup
    def broken_lookup(self, tokens, tenant_id="default", system_prefixes=None):
        return len(tokens)
    pcache.PrefixCache.lookup = broken_lookup
    try:
        out["catches_broken_system_prefix"] = 0.0 if _survives(path) else 1.0
    finally:
        pcache.PrefixCache.lookup = good_lookup
    out["faults_caught"] = out["catches_no_isolation"] + out["catches_broken_system_prefix"]
    return out
