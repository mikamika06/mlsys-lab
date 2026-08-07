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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_block_math": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Learner tests failed on valid code: {e}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import kvplan.planner as p
    good_calc = p.calculate_paged_kv_plan

    def broken_calc(seq_lens, block_size, page_budget):
        return {
            "total_blocks_needed": sum(seq_lens),
            "allocated_blocks": len(seq_lens),
            "blocks_per_seq": seq_lens,
            "waste_tokens": 0,
            "efficiency": 1.0,
            "fits_in_budget": True
        }

    p.calculate_paged_kv_plan = broken_calc
    try:
        out["catches_invalid_block_math"] = 0.0 if _survives(path) else 1.0
    finally:
        p.calculate_paged_kv_plan = good_calc

    return out
