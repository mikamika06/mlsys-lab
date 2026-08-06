import os
import sys
import importlib.util
import numpy as np
import ref

def _run(path):
    spec = importlib.util.spec_from_file_location("test_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"curve_match": 0.0, "has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_obs": 0.0}

    try:
        from pruning.eval import TinyLM, eval_wanda_curve
        rng = np.random.RandomState(42)
        W1, W2 = rng.randn(16, 8), rng.randn(8, 16)
        X = rng.randn(32, 8)
        mod_ref = ref.TinyLM(W1, W2)
        mod_got = TinyLM(W1, W2)
        sparsities = [0.0, 0.25, 0.5]

        c_ref = ref.eval_wanda_curve(mod_ref, X, sparsities)
        c_got = eval_wanda_curve(mod_got, X, sparsities)
        if len(c_ref) == len(c_got) and np.allclose(c_ref, c_got, atol=1e-4):
            out["curve_match"] = 1.0
    except (ImportError, NotImplementedError):
        pass

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
        if first is None:
            out["_note"] = "no test_* functions found"
            return out
        out["has_tests"] = 1.0
        out["passes_on_good"] = 1.0
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {e}"
        return out

    import pruning.sparsegpt
    good_obs = pruning.sparsegpt.obs_prune

    def bad_obs(W, X, s):
        k = int(W.shape[1] * s)
        mask = np.ones_like(W, dtype=bool)
        if k > 0:
            for i in range(W.shape[0]):
                idx = np.argsort(np.abs(W[i]))[:k]
                mask[i, idx] = False
        return W * mask, mask

    pruning.sparsegpt.obs_prune = bad_obs
    try:
        passed = _run(path)
        if not passed:
            out["catches_broken_obs"] = 1.0
    except Exception:
        out["catches_broken_obs"] = 1.0
    finally:
        pruning.sparsegpt.obs_prune = good_obs

    return out
