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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_bad_placement": 0.0,
        "catches_energy_regression": 0.0,
        "faults_caught": 0.0
    }
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import ane_model.transform as trans

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_frac = trans.measure_ane_fraction
    trans.measure_ane_fraction = lambda model, inp: 0.1
    try:
        out["catches_bad_placement"] = 0.0 if _survives(path) else 1.0
    finally:
        trans.measure_ane_fraction = good_frac

    good_energy = trans.measure_energy_per_request
    trans.measure_energy_per_request = lambda model, inp: 50.0
    try:
        out["catches_energy_regression"] = 0.0 if _survives(path) else 1.0
    finally:
        trans.measure_energy_per_request = good_energy

    out["faults_caught"] = out["catches_bad_placement"] + out["catches_energy_regression"]
    return out
