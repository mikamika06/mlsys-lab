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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_leaky_move": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct logic: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import eplb.reconstruct as rec
    good_apply = rec.apply_event

    def leaky_apply(layout, event):
        new_layout = {k: list(v) for k, v in layout.items()}
        if event.get("type") == "move":
            dst = event["dest"]
            exp = event["expert"]
            if dst in new_layout and exp not in new_layout[dst]:
                new_layout[dst].append(exp)
            for k in new_layout:
                new_layout[k].sort()
            return new_layout
        return good_apply(layout, event)

    rec.apply_event = leaky_apply
    try:
        out["catches_leaky_move"] = 0.0 if _survives(path) else 1.0
    finally:
        rec.apply_event = good_apply

    return out
