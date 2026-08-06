import ref
import numpy as np


def check(workdir):
    from grammar.transition import build_transition_matrix

    specs = ref.get_specs()
    out = {"transitions_matched": 0.0}
    ok = 0
    for i, spec in enumerate(specs):
        want = ref.build_transition_matrix(spec)
        got = build_transition_matrix(spec)
        if got is not None and np.allclose(got, want, atol=1e-5):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"spec {i} transition matrix mismatch"
    out["transitions_matched"] = float(ok)
    return out
