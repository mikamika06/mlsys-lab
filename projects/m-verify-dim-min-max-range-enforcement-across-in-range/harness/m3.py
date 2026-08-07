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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_auto_as_dynamic": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"fails on good: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import shapes.verifier as v
    good_resolve = v.resolve_module_signature

    def bad_resolve(dim_specs, observed_shapes):
        resolved = []
        for i, dim in enumerate(dim_specs):
            sizes = [shape[i] for shape in observed_shapes]
            if dim.dim_type in (v.DimType.AUTO, v.DimType.DYNAMIC):
                resolved.append(v.Dim(dim.name, min(sizes), max(sizes), v.DimType.EXPLICIT))
            else:
                for s in sizes:
                    v.verify_range(dim, s)
                resolved.append(v.Dim(dim.name, dim.min_val, dim.max_val, v.DimType.EXPLICIT))
        return resolved

    v.resolve_module_signature = bad_resolve
    try:
        out["catches_auto_as_dynamic"] = 0.0 if _survives(path) else 1.0
    finally:
        v.resolve_module_signature = good_resolve

    return out
