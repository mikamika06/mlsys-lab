from sdpa_pred.eligibility import is_eligible


def predict_backend(dtype: str, is_causal: bool, q_len: int, kv_len: int, head_dim: int, device_cap: tuple) -> str:
    for b in ["flash_attention", "mem_efficient", "math"]:
        if is_eligible(b, dtype, is_causal, q_len, kv_len, head_dim, device_cap):
            return b
    return "math"
