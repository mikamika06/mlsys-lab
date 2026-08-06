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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unaligned_nm_tiles": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on reference code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import sparsity.hardware as hw
    orig_val = hw.validate_nm_tensorcore_alignment

    def broken_val(M, N, K, dtype_bits=16):
        return {
            "valid": True,
            "m_valid": True,
            "n_valid": True,
            "k_valid": True,
            "required_k_multiple": 1,
        }

    hw.validate_nm_tensorcore_alignment = broken_val
    import sparsity.decision as dec
    dec.validate_nm_tensorcore_alignment = broken_val

    try:
        out["catches_unaligned_nm_tiles"] = 0.0 if _survives(path) else 1.0
    finally:
        hw.validate_nm_tensorcore_alignment = orig_val
        dec.validate_nm_tensorcore_alignment = orig_val

    return out
