import sys
import numpy as np
import ref

def check(workdir):
    out = {"imatrix_calculated": 0.0, "weighted_error_lower": 0.0}

    sys.path.insert(0, workdir)
    from gguf_pipeline.quantizer import compute_imatrix, quantize_imatrix, quantize_q4_0, dequantize_q4_0

    np.random.seed(42)
    activations = np.random.randn(100, 32).astype(np.float32)
    activations[:, 0] *= 10.0

    imat = compute_imatrix(activations)
    if imat.shape == (32,) and imat[0] > imat[1]:
        out["imatrix_calculated"] = 1.0

    weights = np.random.randn(32, 32).astype(np.float32)
    res_imat = quantize_imatrix(weights, imat, n_bits=4)
    q4 = quantize_q4_0(weights)
    deq_q4 = dequantize_q4_0(q4["qdata"], q4["scales"])

    err_imat = np.mean(((weights - res_imat["dequantized"]) ** 2) * imat)
    err_standard = np.mean(((weights - deq_q4) ** 2) * imat)

    if err_imat <= err_standard:
        out["weighted_error_lower"] = 1.0

    return out
