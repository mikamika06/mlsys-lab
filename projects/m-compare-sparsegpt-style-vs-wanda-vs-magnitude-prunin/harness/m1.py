import ref
import numpy as np


def check(workdir):
    from prune.methods import compare_methods

    w, X_match, _ = ref.generate_data()
    sparsity = 0.5

    got = compare_methods(w, X_match, sparsity)

    w_mag, m_mag = ref.magnitude_prune(w, sparsity)
    w_wanda, m_wanda = ref.wanda_prune(w, X_match, sparsity)
    w_sgpt, m_sgpt = ref.sparsegpt_prune(w, X_match, sparsity)

    err_mag = ref.evaluate_quality(w, w_mag, X_match)
    err_wanda = ref.evaluate_quality(w, w_wanda, X_match)
    err_sgpt = ref.evaluate_quality(w, w_sgpt, X_match)

    want = {
        "magnitude": err_mag,
        "wanda": err_wanda,
        "sparsegpt": err_sgpt
    }

    out = {"methods_matched": 0.0, "methods": float(len(want))}
    ok = 0
    for name in want:
        if name in got and np.isclose(got[name], want[name], atol=1e-4):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"method {name}: got {got.get(name)}, want {want[name]}"

    out["methods_matched"] = float(ok)
    return out
