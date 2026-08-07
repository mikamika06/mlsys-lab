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
    sys.path.insert(0, workdir)
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_sliding_boundary_fault": 0.0,
        "catches_full_layer_truncation_fault": 0.0,
        "faults_caught": 0.0,
    }
    if not os.path.isfile(path):
        return out

    import kvcache.cache as cache_mod

    try:
        first = _run(path)
    except Exception:
        return out
    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    orig_append = cache_mod.SingleLayerCache.append

    def faulty_sliding_append(self, k, v):
        if self.k is None:
            self.k = k.copy()
            self.v = v.copy()
        else:
            self.k = cache_mod.np.concatenate([self.k, k], axis=0)
            self.v = cache_mod.np.concatenate([self.v, v], axis=0)
        self.total_seen += k.shape[0]
        if self.is_sliding and self.window_size is not None:
            if self.k.shape[0] > self.window_size:
                self.k = self.k[: self.window_size]
                self.v = self.v[: self.window_size]
        return self.k, self.v

    cache_mod.SingleLayerCache.append = faulty_sliding_append
    try:
        out["catches_sliding_boundary_fault"] = 0.0 if _survives(path) else 1.0
    finally:
        cache_mod.SingleLayerCache.append = orig_append

    def faulty_full_truncation_append(self, k, v):
        if self.k is None:
            self.k = k.copy()
            self.v = v.copy()
        else:
            self.k = cache_mod.np.concatenate([self.k, k], axis=0)
            self.v = cache_mod.np.concatenate([self.v, v], axis=0)
        self.total_seen += k.shape[0]
        if self.k.shape[0] > 10:
            self.k = self.k[-10:]
            self.v = self.v[-10:]
        return self.k, self.v

    cache_mod.SingleLayerCache.append = faulty_full_truncation_append
    try:
        out["catches_full_layer_truncation_fault"] = (
            0.0 if _survives(path) else 1.0
        )
    finally:
        cache_mod.SingleLayerCache.append = orig_append

    out["faults_caught"] = (
        out["catches_sliding_boundary_fault"]
        + out["catches_full_layer_truncation_fault"]
    )
    return out
