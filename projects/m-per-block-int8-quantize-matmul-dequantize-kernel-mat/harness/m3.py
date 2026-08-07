import importlib.util
import os
import ref

def _run(path):
    spec = importlib.util.spec_from_file_location("test_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns: return None
    for fn in fns: fn()
    return True

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_matmul": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import qmat
    import qmat.matmul as m_mod
    good_fn = m_mod.per_block_int8_matmul

    def bad_matmul(A, B, block_size):
        sA = max(ref.np.max(ref.np.abs(A)) / 127.0, 1e-9)
        sB = max(ref.np.max(ref.np.abs(B)) / 127.0, 1e-9)
        Aq = ref.np.round(A / sA)
        Bq = ref.np.round(B / sB)
        return ref.np.dot(Aq, Bq) * (sA * sB)

    m_mod.per_block_int8_matmul = bad_matmul
    try:
        survives = False
        try:
            survives = _run(path) is True
        except Exception:
            survives = False
        if not survives:
            out["catches_bad_matmul"] = 1.0
    finally:
        m_mod.per_block_int8_matmul = good_fn

    return out
