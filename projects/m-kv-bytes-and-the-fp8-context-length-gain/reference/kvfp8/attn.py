import numpy as np
from kvfp8.quant import dequantize_fp8_per_head, quantize_fp8_per_head


def compute_attention_error_by_position(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    quantize_fn=None,
    dequantize_fn=None,
) -> np.ndarray:
    if quantize_fn is None:
        quantize_fn = quantize_fp8_per_head
    if dequantize_fn is None:
        dequantize_fn = dequantize_fp8_per_head

    seq_len, num_heads, head_dim = q.shape
    d_k = head_dim

    qk_ref = np.einsum("thd,shd->ths", q, k) / np.sqrt(d_k)
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    qk_ref_masked = np.where(mask[:, None, :], -1e9, qk_ref)

    s_ref = np.exp(qk_ref_masked - np.max(qk_ref_masked, axis=-1, keepdims=True))
    s_ref = s_ref / np.sum(s_ref, axis=-1, keepdims=True)
    out_ref = np.einsum("ths,shd->thd", s_ref, v)

    k_q, k_scale = quantize_fn(k)
    k_deq = dequantize_fn(k_q, k_scale)
    v_q, v_scale = quantize_fn(v)
    v_deq = dequantize_fn(v_q, v_scale)

    qk_q = np.einsum("thd,shd->ths", q, k_deq) / np.sqrt(d_k)
    qk_q_masked = np.where(mask[:, None, :], -1e9, qk_q)

    s_q = np.exp(qk_q_masked - np.max(qk_q_masked, axis=-1, keepdims=True))
    s_q = s_q / np.sum(s_q, axis=-1, keepdims=True)
    out_q = np.einsum("ths,shd->thd", s_q, v_deq)

    num = np.linalg.norm(out_ref - out_q, axis=(-2, -1))
    den = np.linalg.norm(out_ref, axis=(-2, -1)) + 1e-12
    return num / den
