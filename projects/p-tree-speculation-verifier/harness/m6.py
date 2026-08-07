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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "catches_broken_expectation": 0.0, "catches_leaky_mask": 0.0}

    if not os.path.isfile(path):
        return out

    import speculation.tree as tree

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out
    out["has_tests"] = 1.0

    good_el = tree.expected_length

    def broken_el(tokens, parents, draft_probs, target_probs):
        n = len(parents)
        E = [1.0] * n
        for i in range(n - 1, -1, -1):
            children = [j for j, p in enumerate(parents) if p == i]
            E[i] = 1.0 + sum(E[j] for j in children)
        root_children = [j for j, p in enumerate(parents) if p == -1]
        return 1.0 + sum(E[j] for j in root_children)

    tree.expected_length = broken_el
    try:
        out["catches_broken_expectation"] = 0.0 if _survives(path) else 1.0
    finally:
        tree.expected_length = good_el

    good_mask = tree.tree_attention_mask

    def broken_mask(parents):
        import numpy as np
        n = len(parents)
        return np.ones((n, n), dtype=bool)

    tree.tree_attention_mask = broken_mask
    try:
        out["catches_leaky_mask"] = 0.0 if _survives(path) else 1.0
    finally:
        tree.tree_attention_mask = good_mask

    return out
