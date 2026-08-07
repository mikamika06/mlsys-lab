import ref
import math
import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from perf import analyzer
    except ImportError:
        return {"error_match": 0.0, "validate_good": 0.0, "validate_bad": 0.0}

    out = {"error_match": 0.0, "validate_good": 0.0, "validate_bad": 0.0}

    ok_err = 0
    cases = [(100.0, 0.01, 1), (50.0, 0.04, 2), (1000.0, 0.01, 8)]
    for t, l, c in cases:
        want = ref.consistency_error(t, l, c)
        try:
            got = analyzer.consistency_error(t, l, c)
            if math.isclose(want, got, rel_tol=1e-5, abs_tol=1e-9):
                ok_err += 1
            else:
                if "_note_err" not in out:
                    out["_note_err"] = f"for t={t}, l={l}, c={c} got {got} want {want}"
        except Exception:
            pass
    out["error_match"] = float(ok_err)

    good_trace = ref.generate_valid_trace(200, 4, 0.002, 0.020, 0.005)
    bad_trace = ref.generate_inconsistent_trace()

    try:
        err = analyzer.validate_trace(good_trace, 4)
        out["validate_good"] = 1.0 if err < 0.05 else 0.0
    except Exception as e:
        out["_note_good"] = f"validate_trace raised on good trace: {e}"

    try:
        analyzer.validate_trace(bad_trace, 1)
        out["_note_bad"] = "did not raise ValueError on bad trace"
    except ValueError:
        out["validate_bad"] = 1.0
    except Exception as e:
        out["_note_bad"] = f"raised wrong error type: {type(e)}"

    return out
