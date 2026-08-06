import sys
import ref
import numpy as np

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from speculative.model import expected_tokens, compute_speedup
    except ImportError:
        sys.path.pop(0)
        return {"valid": 0.0, "rel_err": 0.0, "_note": "failed to import module"}

    out = {"valid": 1.0, "rel_err": 0.0}
    alphas_list = [
        np.array([0.9, 0.8, 0.7]),
        np.array([0.5, 0.5, 0.5]),
        np.array([0.1, 0.9, 0.1]),
        np.array([0.99, 0.99, 0.99, 0.99])
    ]

    errs = []
    for a in alphas_list:
        try:
            want = ref.expected_tokens(a)
            got = expected_tokens(a)
            if want > 0:
                errs.append(abs(want - got) / want)
        except Exception:
            errs.append(1.0)

    try:
        want_s = ref.compute_speedup(2.5, 4, 0.1)
        got_s = compute_speedup(2.5, 4, 0.1)
        errs.append(abs(want_s - got_s) / want_s)
    except Exception:
        errs.append(1.0)

    if errs and max(errs) < 1e-4:
        out["rel_err"] = 1.0

    sys.path.pop(0)
    return out
