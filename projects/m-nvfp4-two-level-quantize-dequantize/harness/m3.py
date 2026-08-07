import importlib.util
import os
import numpy as np


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_vec_scale": 0.0}
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

    import quant.formats as qf
    good = qf.nvfp4

    def broken_nvfp4(x, block_size=16, super_block=256):
        sup_blocks = x.reshape(-1, super_block)
        m_sup = np.max(np.abs(sup_blocks), axis=1, keepdims=True)
        m_sup = np.maximum(m_sup, 1e-12)
        s_sup = 2.0 ** np.ceil(np.log2(m_sup / 6.0))
        scaled_sup = sup_blocks / s_sup
        q = qf.round_e2m1(scaled_sup)
        return (q * s_sup).reshape(x.shape)

    qf.nvfp4 = broken_nvfp4
    try:
        out["catches_broken_vec_scale"] = 0.0 if _survives(path) else 1.0
    finally:
        qf.nvfp4 = good

    return out
