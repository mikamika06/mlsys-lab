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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_crop": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import prefixcache.dynamic as dyn

    orig_crop = dyn.DynamicCache.crop

    def broken_crop(self, max_length):
        if max_length < 0:
            raise ValueError("max_length must be non-negative")
        if len(self.key_cache) > 0 and self.key_cache[0] is not None:
            cur_len = self.key_cache[0].shape[2]
            if cur_len > max_length:
                self.key_cache[0] = self.key_cache[0][:, :, :max_length, :]
                self.value_cache[0] = self.value_cache[0][:, :, :max_length, :]

    dyn.DynamicCache.crop = broken_crop
    try:
        out["catches_broken_crop"] = 0.0 if _survives(path) else 1.0
    finally:
        dyn.DynamicCache.crop = orig_crop

    return out
