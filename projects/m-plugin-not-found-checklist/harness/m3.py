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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_roundtrip": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on valid reference implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import trtplugin.serialize as ser
    orig_deserialize = ser.deserialize_fields

    def broken_deserialize(data_bytes):
        res = orig_deserialize(data_bytes)
        for k in res:
            if isinstance(res[k], list):
                res[k] = [x + 1 for x in res[k]]
        return res

    ser.deserialize_fields = broken_deserialize
    import trtplugin
    trtplugin.serialize.deserialize_fields = broken_deserialize

    try:
        if _survives(path):
            out["catches_invalid_roundtrip"] = 0.0
            out["_note"] = "regression tests failed to catch corrupted array field deserialization"
        else:
            out["catches_invalid_roundtrip"] = 1.0
    finally:
        ser.deserialize_fields = orig_deserialize
        trtplugin.serialize.deserialize_fields = orig_deserialize

    return out
