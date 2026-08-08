def dtype_size(dtype_str):
    mapping = {
        "float32": 4,
        "fp32": 4,
        "float16": 2,
        "fp16": 2,
        "bfloat16": 2,
        "bf16": 2,
        "int8": 1,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def calculate_eager_memory(batch_size, seq_len, num_heads, head_dim, dtype_str="float16"):
    b = dtype_size(dtype_str)
    b_sz = batch_size
    s = seq_len
    h = num_heads
    d = head_dim

    qkv_bytes = 3 * b_sz * s * h * d * b
    attn_scores_bytes = b_sz * h * s * s * 4
    attn_probs_bytes = b_sz * h * s * s * b
    context_bytes = b_sz * s * h * d * b

    stored_forward_bytes = qkv_bytes + attn_probs_bytes
    fwd_peak_bytes = qkv_bytes + attn_scores_bytes + attn_probs_bytes + context_bytes
    bwd_peak_bytes = stored_forward_bytes + (2 * b_sz * h * s * s * b) + (3 * b_sz * s * h * d * b)

    return {
        "stored_forward_bytes": stored_forward_bytes,
        "fwd_peak_bytes": fwd_peak_bytes,
        "bwd_peak_bytes": bwd_peak_bytes,
    }


def calculate_sdpa_memory(batch_size, seq_len, num_heads, head_dim, dtype_str="float16"):
    b = dtype_size(dtype_str)
    b_sz = batch_size
    s = seq_len
    h = num_heads
    d = head_dim

    qkv_bytes = 3 * b_sz * s * h * d * b
    lse_bytes = b_sz * h * s * 4
    out_bytes = b_sz * s * h * d * b

    stored_forward_bytes = qkv_bytes + lse_bytes
    fwd_peak_bytes = qkv_bytes + lse_bytes + out_bytes
    bwd_peak_bytes = stored_forward_bytes + out_bytes + (3 * b_sz * s * h * d * b)

    return {
        "stored_forward_bytes": stored_forward_bytes,
        "fwd_peak_bytes": fwd_peak_bytes,
        "bwd_peak_bytes": bwd_peak_bytes,
    }
