import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import numpy as np
    out = {"deq_rel_err": 0.0, "matmul_works": 0.0, "get_rel_err_works": 0.0}
    w = ref.generate_weight()

    try:
        from qtensor.subclass import quantize_affine
        q = quantize_affine(w, 32, True)
        deq = q.dequantize()
        err = np.linalg.norm(deq - w) / np.linalg.norm(w)
        
        ref_q = ref.quantize_affine(w, 32, True)
        ref_err = np.linalg.norm(ref_q.dequantize() - w) / np.linalg.norm(w)
        
        if err <= ref_err * 1.5:
            out["deq_rel_err"] = 1.0
    except Exception as e:
        out["_note_m2_deq"] = str(e)

    try:
        from qtensor.subclass import quantize_affine
        q2 = quantize_affine(w, 64, False)
        act = ref.generate_act()
        res = q2 @ act.T
        ref_q2 = ref.quantize_affine(w, 64, False)
        ref_res = ref_q2 @ act.T
        
        err2 = np.linalg.norm(res - ref_res) / (np.linalg.norm(ref_res) + 1e-9)
        if err2 <= 0.1:
            out["matmul_works"] = 1.0
    except Exception as e:
        out["_note_m2_matmul"] = str(e)

    try:
        from qtensor.compare import get_rel_err
        err_edge = get_rel_err(w, "edge_device")
        ref_err_edge = ref.get_rel_err(w, "edge_device")
        if abs(err_edge - ref_err_edge) < 0.05:
            out["get_rel_err_works"] = 1.0
    except Exception as e:
        out["_note_m2_relerr"] = str(e)

    return out
