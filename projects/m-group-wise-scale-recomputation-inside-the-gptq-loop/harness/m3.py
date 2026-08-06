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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_static_scales": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import gptq.loop as l_mod
    import gptq.scales as s_mod
    good_loop = l_mod.gptq_quantize_with_recompute

    def broken_loop(w, h, group_size, bits):
        w_q = w.copy()
        rows, cols = w.shape
        max_val = float(2 ** (bits - 1) - 1)
        invh = np.linalg.inv(h)
        static_scales = s_mod.compute_group_scales(w, group_size, bits)
        for i in range(cols):
            col_w = w_q[:, i]
            g_idx = i // group_size
            scale = static_scales[:, g_idx]
            q = np.round(col_w / scale)
            q = np.clip(q, -max_val, max_val)
            q_w = q * scale
            err = (col_w - q_w) / invh[i, i]
            w_q[:, i] = q_w
            if i + 1 < cols:
                w_q[:, i+1:] -= np.outer(err, invh[i, i+1:])
        return w_q

    l_mod.gptq_quantize_with_recompute = broken_loop
    try:
        out["catches_static_scales"] = 0.0 if _survives(path) else 1.0
    finally:
        l_mod.gptq_quantize_with_recompute = good_loop
    return out
