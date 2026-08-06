import ref
import numpy as np


def check(workdir):
    from grammar.ceiling import compute_acceptance_ceiling

    specs = ref.get_specs()
    gammas = [2, 4, 3]
    ok = 0
    out = {"ceiling_match": 0.0}
    for i, spec in enumerate(specs):
        mat = ref.build_transition_matrix(spec)
        gamma = gammas[i]
        want = ref.compute_acceptance_ceiling(mat, gamma)
        got = compute_acceptance_ceiling(mat, gamma)
        if got is not None and np.isclose(got, want, atol=1e-5):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"spec {i} ceiling got {got}, want {want}"
    if ok == len(specs):
        out["ceiling_match"] = 1.0
    return out
