import numpy as np


def quantize_dequantize(x: np.ndarray, dtype: str) -> np.ndarray:
    """Quantize and immediately dequantize array x along blocks of size 32."""
    orig_shape = x.shape
    x_flat = x.astype(np.float32).reshape(-1, 32)

    if dtype == "f16":
        out = x_flat.astype(np.float16).astype(np.float32)
    elif dtype == "q8_0":
        max_abs = np.abs(x_flat).max(axis=1, keepdims=True)
        scales = max_abs / 127.0
        scales[scales == 0] = 1.0
        scales_f16 = scales.astype(np.float16).astype(np.float32)
        q = np.clip(np.round(x_flat / scales_f16), -128, 127).astype(np.int8)
        out = q.astype(np.float32) * scales_f16
    elif dtype == "q4_0":
        max_abs = np.abs(x_flat).max(axis=1, keepdims=True)
        scales = max_abs / 7.0
        scales[scales == 0] = 1.0
        scales_f16 = scales.astype(np.float16).astype(np.float32)
        q = np.clip(np.round(x_flat / scales_f16) + 8, 0, 15).astype(np.uint8)
        out = (q.astype(np.float32) - 8.0) * scales_f16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return out.reshape(orig_shape)


def eval_needle_retrieval(
    seq_len: int = 512,
    head_dim: int = 64,
    needle_idx: int = 128,
    dtype: str = "q4_0",
    seed: int = 42,
) -> dict:
    """Run needle-in-haystack retrieval evaluation on quantized KV cache."""
    rng = np.random.RandomState(seed)
    keys = rng.randn(seq_len, head_dim).astype(np.float32)
    vals = rng.randn(seq_len, head_dim).astype(np.float32)

    needle_k = rng.randn(head_dim).astype(np.float32)
    needle_k = 5.0 * needle_k / np.linalg.norm(needle_k)
    needle_v = rng.randn(head_dim).astype(np.float32)

    keys[needle_idx] = needle_k
    vals[needle_idx] = needle_v

    query = needle_k + rng.randn(head_dim).astype(np.float32) * 0.05

    scores_ref = np.dot(keys, query) / np.sqrt(head_dim)
    weights_ref = np.exp(scores_ref - np.max(scores_ref))
    weights_ref /= np.sum(weights_ref)
    out_ref = np.dot(weights_ref, vals)

    k_q = quantize_dequantize(keys, dtype)
    v_q = quantize_dequantize(vals, dtype)

    scores_q = np.dot(k_q, query) / np.sqrt(head_dim)
    weights_q = np.exp(scores_q - np.max(scores_q))
    weights_q /= np.sum(weights_q)
    out_q = np.dot(weights_q, v_q)

    retrieved_idx = int(np.argmax(weights_q))
    needle_found = retrieved_idx == needle_idx

    dot = np.dot(out_ref, out_q)
    norm_ref = np.linalg.norm(out_ref)
    norm_q = np.linalg.norm(out_q)
    cos_sim = float(dot / (norm_ref * norm_q + 1e-9))

    l2_diff = np.linalg.norm(out_ref - out_q)
    rel_err = float(l2_diff / (norm_ref + 1e-9))

    return {
        "retrieved_idx": retrieved_idx,
        "needle_found": needle_found,
        "cosine_similarity": cos_sim,
        "rel_l2_error": rel_err,
    }
