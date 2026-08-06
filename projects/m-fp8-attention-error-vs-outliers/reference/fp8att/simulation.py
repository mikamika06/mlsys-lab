import numpy as np
from fp8att.hadamard import apply_hadamard, generate_hadamard
from fp8att.quant import compute_rel_error, dequantize_fp8, quantize_fp8


def simulate_attention_error(q, k, v):
    scale_q = np.max(np.abs(q)) / 448.0 + 1e-8
    scale_k = np.max(np.abs(k)) / 448.0 + 1e-8
    q_q = dequantize_fp8(quantize_fp8(q, scale_q), scale_q)
    k_q = dequantize_fp8(quantize_fp8(k, scale_k), scale_k)
    scores_ref = np.matmul(q, k.T) / np.sqrt(q.shape[-1])
    scores_fp8 = np.matmul(q_q, k_q.T) / np.sqrt(q.shape[-1])
    err_direct = compute_rel_error(scores_ref, scores_fp8)

    q_h = apply_hadamard(q)
    k_h = apply_hadamard(k)
    scale_qh = np.max(np.abs(q_h)) / 448.0 + 1e-8
    scale_kh = np.max(np.abs(k_h)) / 448.0 + 1e-8
    qh_q = dequantize_fp8(quantize_fp8(q_h, scale_qh), scale_qh)
    kh_q = dequantize_fp8(quantize_fp8(k_h, scale_kh), scale_kh)
    h = generate_hadamard(q.shape[-1])
    scores_fp8_h = np.matmul(np.matmul(qh_q, h.T), np.matmul(kh_q, h.T).T) / np.sqrt(q.shape[-1])
    err_hadamard = compute_rel_error(scores_ref, scores_fp8_h)

    return {"err_direct": err_direct, "err_hadamard": err_hadamard}
