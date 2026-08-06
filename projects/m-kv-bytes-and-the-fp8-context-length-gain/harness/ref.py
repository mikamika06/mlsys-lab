import numpy as np

CONFIGS = [
    {
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "seq_len": 4096,
        "batch_size": 2,
        "gpu_mem": 80 * 1024**3,
        "weights": 14 * 1024**3,
        "act_budget": 2 * 1024**3,
    },
    {
        "num_layers": 80,
        "num_kv_heads": 8,
        "head_dim": 128,
        "seq_len": 8192,
        "batch_size": 1,
        "gpu_mem": 80 * 1024**3,
        "weights": 70 * 1024**3,
        "act_budget": 4 * 1024**3,
    },
    {
        "num_layers": 40,
        "num_kv_heads": 16,
        "head_dim": 128,
        "seq_len": 16384,
        "batch_size": 4,
        "gpu_mem": 40 * 1024**3,
        "weights": 20 * 1024**3,
        "act_budget": 2 * 1024**3,
    },
]


def ref_compute_kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    fp8: bool = False,
) -> int:
    element_size = 1 if fp8 else 2
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * element_size
    return int(batch_size * seq_len * bytes_per_token)


def ref_max_context_length(
    gpu_memory_bytes: int,
    model_weight_bytes: int,
    activation_budget_bytes: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int = 1,
    fp8: bool = False,
) -> int:
    avail_bytes = gpu_memory_bytes - model_weight_bytes - activation_budget_bytes
    if avail_bytes <= 0:
        return 0
    element_size = 1 if fp8 else 2
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * element_size
    bytes_per_seq_pos = batch_size * bytes_per_token
    return int(avail_bytes // bytes_per_seq_pos)


def ref_quantize_fp8_per_head(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_fp8 = 448.0
    eps = 1e-12
    max_vals = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = np.maximum(max_vals / max_fp8, eps)
    q = np.clip(np.round(x / scale), -448.0, 448.0)
    return q, scale


def ref_dequantize_fp8_per_head(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q * scale


def ref_compute_attention_error_by_position(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    quantize_fn=None,
    dequantize_fn=None,
) -> np.ndarray:
    if quantize_fn is None:
        quantize_fn = ref_quantize_fp8_per_head
    if dequantize_fn is None:
        dequantize_fn = ref_dequantize_fp8_per_head

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
