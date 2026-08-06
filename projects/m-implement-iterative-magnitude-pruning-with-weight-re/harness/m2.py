import ref
import numpy as np


def check(workdir):
    from prune.sweep import sparsity_sweep
    np.random.seed(42)
    init_weights = np.random.randn(20)
    sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]

    want = ref.run_reference_sweep(init_weights, sparsities)
    try:
        got = sparsity_sweep(ref.dummy_eval_fn, init_weights, sparsities)
    except Exception as e:
        return {"sweep_matched": 0.0, "_note": f"raised exception: {e}"}

    keys_match = set(want.keys()) == set(got.keys())
    vals_match = keys_match and all(np.isclose(want[k], got[k]) for k in sparsities)

    match = 1.0 if vals_match else 0.0
    out = {"sweep_matched": match}
    if match == 0.0:
        out["_note"] = "sparsity sweep results do not match reference"
    return out
