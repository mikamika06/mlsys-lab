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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_grouping": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct partitioner: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    try:
        import delegate_measure.partitioner as p
    except ImportError:
        return out

    good_x = p.partition_xnnpack

    def bad_xnnpack(ops):
        total_flops = float(sum(o.get("flops", 0.0) for o in ops))
        blob = f"backend:XNNPACK;ops:{len(ops)};flops:{total_flops}".encode("utf-8")
        return [{"opcode": "DELEGATE", "blob": blob}]

    p.partition_xnnpack = bad_xnnpack

    try:
        if not _survives(path):
            out["catches_invalid_grouping"] = 1.0
    finally:
        p.partition_xnnpack = good_x

    return out
