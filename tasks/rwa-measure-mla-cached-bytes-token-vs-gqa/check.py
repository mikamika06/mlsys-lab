import numpy as np

def grade(sol, fx) -> dict:
    # Oracle parameters
    n_kv_heads = 8
    head_dim   = 64
    kv_lora_rank = 32
    dtype      = np.float16

    bytes_per_element = np.dtype(dtype).itemsize
    expected_mla = kv_lora_rank * bytes_per_element
    expected_gqa = 2 * n_kv_heads * head_dim * bytes_per_element

    try:
        got = sol.cached_bytes_per_token(
            n_kv_heads, head_dim, kv_lora_rank, dtype
        )
    except Exception:
        return {"exact_match": 0.0, "size_ratio": float("inf")}

    if not isinstance(got, (tuple, list)) or len(got) != 2:
        return {"exact_match": 0.0, "size_ratio": float("inf")}

    mla_bytes, gqa_bytes = got
    exact = 1.0 if (mla_bytes, gqa_bytes) == (expected_mla, expected_gqa) else 0.0

    # Ratio error
    ratio_err = abs(
        (mla_bytes / gqa_bytes) - (expected_mla / expected_gqa)
    )
    return {"exact_match": exact, "size_ratio": ratio_err}
