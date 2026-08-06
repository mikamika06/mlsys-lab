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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_generated": 0.0}
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

    import router.admission as a
    good_admit = a.admit

    def bad_admit(queue, active, max_total_tokens, max_prefill_tokens, waiting_served_ratio):
        if not queue:
            return []
        if active and len(queue) <= waiting_served_ratio * len(active):
            return []
        admitted_ids = []
        prefill_sum = 0
        active_sum = sum(req["input_len"] for req in active)
        for req in queue:
            if prefill_sum + req["input_len"] > max_prefill_tokens:
                break
            if active_sum + prefill_sum + req["input_len"] > max_total_tokens:
                break
            admitted_ids.append(req["id"])
            prefill_sum += req["input_len"]
        return admitted_ids

    a.admit = bad_admit

    try:
        if not _survives(path):
            out["catches_missing_generated"] = 1.0
        else:
            out["_note"] = "test did not fail when the admission logic ignored generated_len"
    finally:
        a.admit = good_admit

    return out
