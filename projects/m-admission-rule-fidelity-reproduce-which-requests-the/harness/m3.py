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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_capacity_violation": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import vllmsched.admission as adm
    good_admission = adm.simulate_admission

    def broken_admission(requests, max_num_seqs, max_num_batched_tokens, max_model_len):
        schedule = []
        waiting = [dict(r) for r in requests]
        running = []

        while waiting or running:
            admitted_this_step = []
            current_batched_tokens = 0
            current_seqs = 0

            for req in running:
                current_seqs += 1
                current_batched_tokens += 1
                admitted_this_step.append(req["id"])

            next_waiting = []
            for req in waiting:
                req_prompt_len = req["prompt_len"]
                if req_prompt_len > max_model_len:
                    continue

                if current_seqs + 1 <= max_num_seqs:
                    current_seqs += 1
                    current_batched_tokens += req_prompt_len
                    admitted_this_step.append(req["id"])
                    running.append(req)
                else:
                    next_waiting.append(req)

            waiting = next_waiting

            next_running = []
            for req in running:
                req["remaining_output"] -= 1
                if req["remaining_output"] > 0:
                    next_running.append(req)
            running = next_running

            if admitted_this_step:
                schedule.append(admitted_this_step)

        return schedule

    adm.simulate_admission = broken_admission
    import vllmsched
    vllmsched.simulate_admission = broken_admission

    try:
        out["catches_capacity_violation"] = 0.0 if _survives(path) else 1.0
    finally:
        adm.simulate_admission = good_admission
        vllmsched.simulate_admission = good_admission

    return out
