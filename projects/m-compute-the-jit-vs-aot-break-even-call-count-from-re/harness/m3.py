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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ceiling_bug": 0.0}
    sys.path.insert(0, workdir)
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

    import aotbreak.breakeven as be
    good_compute = be.compute_breakeven

    def floor_bug_compute(profile):
        res = good_compute(profile)
        j_comp = profile["jit_compile_ms"]
        j_exec = profile["jit_exec_ms"]
        a_load = profile["aot_load_ms"]
        a_exec = profile["aot_exec_ms"]
        if j_exec < a_exec and (j_comp - a_load) > 0:
            setup_diff = j_comp - a_load
            exec_diff = a_exec - j_exec
            fl_calls = max(1, int(setup_diff / exec_diff))
            res["break_even_calls"] = float(fl_calls)
            res["crossover_latency_ms"] = float(j_comp + fl_calls * j_exec)
        return res

    be.compute_breakeven = floor_bug_compute
    import aotbreak
    aotbreak.breakeven.compute_breakeven = floor_bug_compute

    try:
        out["catches_ceiling_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        be.compute_breakeven = good_compute
        aotbreak.breakeven.compute_breakeven = good_compute

    return out
