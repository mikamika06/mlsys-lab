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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_usp": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import seqcomm.formulas as f
    good_usp = f.usp_comm_volume_per_layer

    def broken_usp(seq_len, hidden_dim, world_size, ulysses_degree, ring_degree, dtype_bytes=2):
        return ulysses_degree * ring_degree * seq_len * hidden_dim

    f.usp_comm_volume_per_layer = broken_usp
    import seqcomm
    seqcomm.formulas.usp_comm_volume_per_layer = broken_usp

    try:
        out["catches_invalid_usp"] = 0.0 if _survives(path) else 1.0
    finally:
        f.usp_comm_volume_per_layer = good_usp
        seqcomm.formulas.usp_comm_volume_per_layer = good_usp

    return out
