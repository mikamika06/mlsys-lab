import ref
import sys

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from admission.sim import simulate
    except ImportError:
        return {"_note": "could not import admission.sim.simulate"}

    ok = 0
    out = {"traces_matched": 0.0}

    for i, trace in enumerate(ref.TRACES):
        configs = [
            ("accept_all", {}),
            ("queue_limit", {"max_len": 5}),
            ("time_limit", {"max_wait": 15})
        ]
        for pol, kwargs in configs:
            want = ref.simulate(trace, pol, **kwargs)
            try:
                got = simulate(trace, pol, **kwargs)
            except NotImplementedError:
                return out
            except Exception as e:
                out["_note"] = f"crash on trace {i} pol {pol}: {e}"
                return out

            if want == got:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"mismatch on trace {i} pol {pol}: got {got[:2]} want {want[:2]}"

    out["traces_matched"] = float(ok)
    return out
