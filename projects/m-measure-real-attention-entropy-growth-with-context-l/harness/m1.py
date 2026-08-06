import sys
import numpy as np
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from longctx_eval.entropy import compute_attention_entropies
    except ImportError:
        return {"rel_err": 1.0, "_note": "could not import module"}

    q, k = ref.get_m1_data()
    want = ref.compute_attention_entropies(q, k)

    try:
        got = compute_attention_entropies(q, k)
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"crashed: {e}"}

    if not isinstance(got, np.ndarray) or got.shape != want.shape:
        return {"rel_err": 1.0, "_note": "output shape mismatch"}

    diff = np.abs(want - got)
    rel_err = float(np.max(diff / (np.abs(want) + 1e-9)))
    return {"rel_err": rel_err}
