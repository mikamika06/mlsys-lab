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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_flawed_decomposition": 0.0}
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

    import converter.rewrite as rw
    good_apply = rw.apply_decomposition_table

    def broken_apply(graph_spec, equivalence_table):
        res = good_apply(graph_spec, equivalence_table)
        for node in res.get("nodes", []):
            node["op_type"] = "INVALID_BROKEN_OP"
        return res

    rw.apply_decomposition_table = broken_apply
    import converter
    converter.rewrite.apply_decomposition_table = broken_apply

    try:
        out["catches_flawed_decomposition"] = 0.0 if _survives(path) else 1.0
    finally:
        rw.apply_decomposition_table = good_apply
        converter.rewrite.apply_decomposition_table = good_apply

    return out
