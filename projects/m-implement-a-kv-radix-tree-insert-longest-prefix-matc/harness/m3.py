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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_match": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Learner test suite failed on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import kvradix.radix as r
    orig_match = r.RadixTree.match_prefix

    def broken_match(self, tokens):
        matched_len, node, rem = orig_match(self, tokens)
        return (matched_len // 2), node, tokens[(matched_len // 2):]

    r.RadixTree.match_prefix = broken_match
    import kvradix
    kvradix.radix.RadixTree.match_prefix = broken_match

    try:
        survived = _survives(path)
        out["catches_broken_match"] = 0.0 if survived else 1.0
    finally:
        r.RadixTree.match_prefix = orig_match
        kvradix.radix.RadixTree.match_prefix = orig_match

    return out
