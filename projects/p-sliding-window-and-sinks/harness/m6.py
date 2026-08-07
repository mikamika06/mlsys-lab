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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_evicted_sinks": 0.0,
        "catches_broken_window_indexing": 0.0,
        "faults_caught": 0.0,
    }

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    import attn.cache as cache_mod
    import attn.window_sink as ws_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on valid reference: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    orig_append = cache_mod.WindowSinkKVCache.append

    def leaky_append(self, k, v):
        k = cache_mod.np.asarray(k, dtype=cache_mod.np.float64)
        v = cache_mod.np.asarray(v, dtype=cache_mod.np.float64)
        for i in range(len(k)):
            self.k_window.append(k[i : i + 1])
            self.v_window.append(v[i : i + 1])
            if len(self.k_window) > self.window_size:
                self.k_window.pop(0)
                self.v_window.pop(0)
        self.k_sinks = None
        self.v_sinks = None

    cache_mod.WindowSinkKVCache.append = leaky_append
    try:
        out["catches_evicted_sinks"] = 0.0 if _survives(path) else 1.0
    finally:
        cache_mod.WindowSinkKVCache.append = orig_append

    orig_attn = ws_mod.compute_window_sink_attention

    def broken_window_attn(q, k, v, num_sinks, window_size):
        L, d = q.shape
        out_arr = ws_mod.np.zeros_like(q)
        for i in range(L):
            start = max(0, i - window_size + 1)
            sub_k = k[start : i + 1]
            sub_v = v[start : i + 1]
            qi = q[i : i + 1]
            scores = ws_mod.np.dot(qi, sub_k.T) / ws_mod.np.sqrt(d)
            w = ws_mod.np.exp(scores) / ws_mod.np.sum(
                ws_mod.np.exp(scores), axis=-1, keepdims=True
            )
            out_arr[i] = ws_mod.np.dot(w, sub_v)[0]
        return out_arr

    ws_mod.compute_window_sink_attention = broken_window_attn
    try:
        out["catches_broken_window_indexing"] = 0.0 if _survives(path) else 1.0
    finally:
        ws_mod.compute_window_sink_attention = orig_attn

    out["faults_caught"] = (
        out["catches_evicted_sinks"] + out["catches_broken_window_indexing"]
    )
    return out
