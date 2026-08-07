import importlib.util
import os
import sys


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_repack": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import repack.pack as pack_mod
    import repack.permute as perm_mod

    orig_repack = pack_mod.repack_gptq_to_marlin

    def broken_repack(packed_gptq, K, N):
        orig_perm = perm_mod.ROW_PERM
        perm_mod.ROW_PERM = list(range(16))
        try:
            res = orig_repack(packed_gptq, K, N)
            return res
        finally:
            perm_mod.ROW_PERM = orig_perm

    pack_mod.repack_gptq_to_marlin = broken_repack
    try:
        out["catches_broken_repack"] = 0.0 if _survives(path) else 1.0
    finally:
        pack_mod.repack_gptq_to_marlin = orig_repack

    return out
