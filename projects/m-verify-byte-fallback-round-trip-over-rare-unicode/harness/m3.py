import importlib.util
import os
import sys


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_naive_decoder": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import bytefallback.convert as conv
    good_decode = conv.decode_with_fallback

    def naive_decode(token_ids, inv_vocab):
        res = []
        for tid in token_ids:
            tok = inv_vocab.get(tid, "")
            if tok.startswith("<0x") and tok.endswith(">") and len(tok) == 6:
                b_val = int(tok[3:5], 16)
                res.append(bytes([b_val]).decode("utf-8", errors="replace"))
            else:
                res.append(tok)
        return "".join(res)

    conv.decode_with_fallback = naive_decode

    try:
        survived = _survives(path)
        out["catches_naive_decoder"] = 0.0 if survived else 1.0
    finally:
        conv.decode_with_fallback = good_decode

    return out
