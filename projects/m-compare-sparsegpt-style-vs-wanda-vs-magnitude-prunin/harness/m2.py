import ref
import numpy as np


def check(workdir):
    from prune.debug import debug_wanda_domain

    w, X_match, X_mismatch = ref.generate_data()
    sparsity = 0.5

    got = debug_wanda_domain(w, X_match, X_mismatch, sparsity)
    want = ref.diagnose_domain_mismatch(w, X_match, X_mismatch, sparsity)

    out = {"domain_debug_match": 0.0}
    match = True
    for k in want:
        if isinstance(want[k], float):
            if not np.isclose(got.get(k, -999), want[k], atol=1e-4):
                match = False
        else:
            if got.get(k) != want[k]:
                match = False

    if match:
        out["domain_debug_match"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"

    return out
