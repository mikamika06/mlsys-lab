import importlib.util
import os
import sys

sys.path.insert(0, ".")


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
        "catches_top_down_wrapping": 0.0,
    }

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"tests failed on reference implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import fsdp_verify.reconstruct as rec

    good_func = rec.reconstruct_fully_shard_sequence

    def top_down_broken_reconstruct(tree):
        seq = []

        def pre_order(node, path):
            if node.get("should_shard", True):
                seq.append(path)
            for child_name, child_node in node.get("children", {}).items():
                child_path = f"{path}.{child_name}" if path else child_name
                pre_order(child_node, child_path)

        pre_order(tree, "")
        return seq

    rec.reconstruct_fully_shard_sequence = top_down_broken_reconstruct
    try:
        out["catches_top_down_wrapping"] = 0.0 if _survives(path) else 1.0
    finally:
        rec.reconstruct_fully_shard_sequence = good_func

    return out
