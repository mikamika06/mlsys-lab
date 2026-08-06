import os
import importlib.util

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

def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False

def check(workdir):
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"exact_match": 0.0}
    if not os.path.isfile(path):
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["_note"] = f"fails on good implementation: {e}"
        return out

    if not first:
        return out

    import sys
    sys.path.insert(0, workdir)
    import shapes.verifier as sv
    good = sv.propagate_shapes

    def broken(ops, inputs, constraints):
        res = good(ops, inputs, constraints)
        if ops:
            last_out = ops[-1]["out"]
            shape = list(res[last_out])
            shape[-1] = (999, None)
            res[last_out] = tuple(shape)
        return res

    sv.propagate_shapes = broken
    try:
        survives = _survives(path)
        if not survives:
            out["exact_match"] = 1.0
    finally:
        sv.propagate_shapes = good

    return out
