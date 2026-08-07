import ref
import numpy as np

def check(workdir):
    from headprune.importance import compute_importance
    rng = np.random.default_rng(42)
    acts = rng.normal(size=(10, 4, 16, 32))
    grads = rng.normal(size=(10, 4, 16, 32))
    want = ref.compute_importance(acts, grads)
    got = compute_importance(acts, grads)
    rel_err = float(np.max(np.abs(got - want) / (np.abs(want) + 1e-8)))
    return {"rel_err": rel_err}
