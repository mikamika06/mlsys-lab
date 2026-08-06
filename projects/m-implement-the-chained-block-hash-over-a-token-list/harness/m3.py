import importlib.util
import os
import hashlib

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unchained": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import blockhash.hashing as h
    good = h.block_hashes

    def unchained_hashes(tokens, block_size):
        out_ = []
        for i in range(0, (len(tokens) // block_size) * block_size, block_size):
            chunk = tokens[i:i+block_size]
            s = ",".join(map(str, chunk)).encode("utf-8")
            digest = hashlib.sha256(s).digest()
            out_.append(digest.hex())
        return out_

    h.block_hashes = unchained_hashes
    try:
        if not _survives(path):
            out["catches_unchained"] = 1.0
        else:
            out["_note"] = "tests passed even when hashing was unchained"
    finally:
        h.block_hashes = good

    return out
