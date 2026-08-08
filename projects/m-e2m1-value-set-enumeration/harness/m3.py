import importlib.util
import os
import math

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

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_no_subnormal": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    try:
        import microscale.e2m1 as e2
    except ImportError:
        return out

    good_decode = e2.decode_e2m1

    def bad_decode(val, bias, has_nan, has_inf):
        # INJECTED BUG: completely ignores the special subnormal / 0.0 rule for E=0.
        # This will evaluate E=0 using the normal formula and break invariants.
        s = (val >> 3) & 1
        e = (val >> 1) & 3
        m = val & 1
        sign = -1.0 if s == 1 else 1.0

        if e == 3:
            if has_nan and m == 1:
                return float('nan')
            if has_inf and m == 0:
                return float('inf') * sign

        # BUG: evaluates e=0 identically to e=1 but with E=0 exponent.
        return sign * (2.0 ** (e - bias)) * (1.0 + m * 0.5)

    e2.decode_e2m1 = bad_decode
    try:
        survives = False
        try:
            survives = _run(path) is True
        except Exception:
            survives = False

        out["catches_no_subnormal"] = 0.0 if survives else 1.0
    finally:
        e2.decode_e2m1 = good_decode

    if out["catches_no_subnormal"] == 0.0:
        out["_note"] = "tests passed even when the subnormal branch (0.0 logic) was silently ignored"

    return out
