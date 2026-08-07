import ref
import numpy as np

def check(workdir):
    try:
        from compression.pipeline import find_interaction_flaw
    except ImportError:
        return {"m2_ok": 0.0}

    w = ref.generate_fixture()
    try:
        res = find_interaction_flaw(w, 0.5, 4)
    except Exception:
        return {"m2_ok": 0.0}

    ref_pq = ref.quantize(ref.prune(w, 0.5), 4)
    ref_qp = ref.prune(ref.quantize(w, 4), 0.5)
    mse_pq = float(np.mean((w - ref_pq)**2))
    mse_qp = float(np.mean((w - ref_qp)**2))

    if abs(res.get("mse_pq", 0) - mse_pq) < 1e-5 and abs(res.get("mse_qp", 0) - mse_qp) < 1e-5:
        return {"m2_ok": 1.0}
    return {"m2_ok": 0.0}
