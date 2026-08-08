import importlib.util
import os

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignorant_cutoff": 0.0}
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

    import speculative_quant.cutoff as c
    good_find = c.find_int8_cutoff

    def mutant_find(draft_sizes, s_target, K, mem_bw, alphas_fp16, alphas_int8, overheads):
        for s in sorted(draft_sizes):
            t_fp16 = (s * 2) / mem_bw + overheads["draft_fp16"]
            t_int8 = (s * 1) / mem_bw + overheads["draft_int8"]
            if t_int8 < t_fp16:
                return s
        return None

    c.find_int8_cutoff = mutant_find

    try:
        survived = _survives(path)
        if survived:
            out["_note"] = "test did not fail when cutoff algorithm only compared generation time instead of overall throughput"
        out["catches_ignorant_cutoff"] = 0.0 if survived else 1.0
    finally:
        c.find_int8_cutoff = good_find

    return out
