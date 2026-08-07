import ref
import numpy as np

def check(workdir):
    try:
        from compression.pipeline import measure_both_orders
    except ImportError:
        return {"m1_ok": 0.0}

    w = ref.generate_fixture()
    try:
        pq, qp = measure_both_orders(w, 0.4, 4)
    except Exception:
        return {"m1_ok": 0.0}

    ref_pq = ref.quantize(ref.prune(w, 0.4), 4)
    ref_qp = ref.prune(ref.quantize(w, 4), 0.4)

    if np.allclose(pq, ref_pq, atol=1e-5) and np.allclose(qp, ref_qp, atol=1e-5):
        return {"m1_ok": 1.0}
    return {"m1_ok": 0.0}
