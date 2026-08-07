import ref
import numpy as np

def check(workdir):
    from qcache.error import compute_quant_error
    np.random.seed(42)
    tensor = np.random.randn(16, 16)
    err4 = compute_quant_error(tensor, 4)
    err2 = compute_quant_error(tensor, 2)
    ref_err4 = ref.ref_compute_quant_error(tensor, 4)
    ref_err2 = ref.ref_compute_quant_error(tensor, 2)
    match = 1.0 if (abs(err4 - ref_err4) < 1e-5 and abs(err2 - ref_err2) < 1e-5 and err2 > err4) else 0.0
    return {"error_match": match}
