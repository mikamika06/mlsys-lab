import importlib.util
import os
import sys


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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_unoptimized_graph": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"The tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ipexaudit.graph as g

    good_diff = g.diff_op_graphs

    def broken_diff(manual_graph, ipex_graph):
        res = good_diff(manual_graph, ipex_graph)
        res["is_ipex_optimized"] = True
        res["redundant_copies_removed"] = 0
        return res

    g.diff_op_graphs = broken_diff
    import ipexaudit

    ipexaudit.graph.diff_op_graphs = broken_diff

    try:
        out["catches_unoptimized_graph"] = 0.0 if _survives(path) else 1.0
    finally:
        g.diff_op_graphs = good_diff
        ipexaudit.graph.diff_op_graphs = good_diff

    return out
