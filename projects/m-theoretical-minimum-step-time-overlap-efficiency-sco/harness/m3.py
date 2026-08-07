import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
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


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_fake_saturation": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct reference implementation: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import overlap.saturation as sat

    good_fn = sat.find_saturation_point

    def fake_find_saturation(bucket_profiles):
        if not bucket_profiles:
            return {"saturation_bucket_mb": 0, "min_step_time": 0.0}
        return {
            "saturation_bucket_mb": bucket_profiles[0]["bucket_size_mb"],
            "min_step_time": bucket_profiles[0]["total_step_time"],
        }

    sat.find_saturation_point = fake_find_saturation
    import overlap

    overlap.find_saturation_point = fake_find_saturation

    try:
        out["catches_fake_saturation"] = 0.0 if _survives(path) else 1.0
    finally:
        sat.find_saturation_point = good_fn
        overlap.find_saturation_point = good_fn

    return out
