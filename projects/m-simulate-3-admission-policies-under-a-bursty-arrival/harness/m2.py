import ref
import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from admission.sim import find_trigger
    except ImportError:
        return {"_note": "could not import find_trigger"}

    ok = 0
    out = {"triggers_matched": 0.0}

    for i, trace in enumerate(ref.TRACES):
        want = ref.find_trigger(trace, 15)
        try:
            got = find_trigger(trace, 15)
        except NotImplementedError:
            return out
        except Exception as e:
            out["_note"] = f"crash on trace {i}: {e}"
            return out

        if want == got:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"mismatch on trace {i}: got {got} want {want}"

    out["triggers_matched"] = float(ok)
    return out
