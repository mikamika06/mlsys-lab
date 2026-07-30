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


def _broken_simulate(n_ctx, n_keep, n_tokens):
    resident = []
    evicted_all = []
    shift_events = []
    next_id = 0
    for _ in range(n_tokens):
        evicted = []
        if len(resident) + 1 > n_ctx:
            n_left = len(resident) - n_keep
            if n_left < 0:
                n_left = 0
            n_discard = n_left // 2
            evicted = resident[:n_discard]
            resident = resident[n_discard:]
        resident.append(next_id)
        next_id += 1
        evicted_all.extend(evicted)
        shift_events.append(len(evicted))
    return {"resident": resident, "evicted": evicted_all, "shift_events": shift_events}


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_leaked_keep": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ctxshift.bookkeeping as bk
    good = bk.simulate

    bk.simulate = _broken_simulate
    import ctxshift
    ctxshift.simulate = _broken_simulate
    try:
        out["catches_leaked_keep"] = 0.0 if _survives(path) else 1.0
    finally:
        bk.simulate = good
        ctxshift.simulate = good
    return out
