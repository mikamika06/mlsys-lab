import sys
import numpy as np
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from pruning.sparsegpt import obs_prune
    except ImportError:
        return {"_note": "failed to import pruning.sparsegpt"}

    rng = np.random.RandomState(1337)
    W = rng.randn(8, 32)
    X = rng.randn(64, 32)

    out = {"w_match": 0.0, "m_match": 0.0}
    try:
        w_ref, m_ref = ref.obs_prune(W, X, 0.5)
        w_got, m_got = obs_prune(W, X, 0.5)

        w_err = np.max(np.abs(w_ref - w_got))
        if w_err < 1e-4:
            out["w_match"] = 1.0

        if np.array_equal(m_ref, m_got):
            out["m_match"] = 1.0

    except NotImplementedError:
        pass

    return out
