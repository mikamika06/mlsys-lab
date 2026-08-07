import importlib.util
import os
import sys


def _run_tests(workdir):
    test_path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(test_path):
        return None
    spec = importlib.util.spec_from_file_location("learner_regression", test_path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, workdir)
    try:
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
    finally:
        if workdir in sys.path:
            sys.path.remove(workdir)


def check(workdir):
    out = {"rel_err": 1.0}
    test_path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(test_path):
        return out

    try:
        first_pass = _run_tests(workdir)
    except Exception:
        return out

    if first_pass is not True:
        return out

    import router.prefix as pref_mod
    import router.bakeoff as bake_mod

    orig_match = pref_mod.compute_prefix_match

    def broken_prefix_match(req_blocks, worker_blocks):
        return 0

    pref_mod.compute_prefix_match = broken_prefix_match
    bake_mod.compute_prefix_match = broken_prefix_match

    try:
        fault_caught = False
        try:
            _run_tests(workdir)
        except Exception:
            fault_caught = True

        if fault_caught:
            out["rel_err"] = 0.0
    finally:
        pref_mod.compute_prefix_match = orig_match
        bake_mod.compute_prefix_match = orig_match

    return out
