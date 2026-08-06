def kv_cache_bytes(prompt_len: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    return 2 * num_layers * num_kv_heads * head_dim * prompt_len * dtype_bytes


def prefill_flops(prompt_len: int, model_cfg: dict) -> int:
    l = model_cfg["num_layers"]
    h = model_cfg["hidden_dim"]
    hq = model_cfg["num_heads"] * model_cfg["head_dim"]
    hkv = model_cfg["num_kv_heads"] * model_cfg["head_dim"]
    d_int = model_cfg["intermediate_dim"]
    gemm_flops_per_tok = 2 * h * (2 * hq + 2 * hkv) + 6 * h * d_int
    attn_flops = 2 * l * model_cfg["num_heads"] * model_cfg["head_dim"] * (prompt_len ** 2)
    return l * gemm_flops_per_tok * prompt_len + attn_flops


def prefill_time_ms(prompt_len: int, model_cfg: dict, tflops: float) -> float:
    flops = prefill_flops(prompt_len, model_cfg)
    return (flops / (tflops * 1e12)) * 1000.0


def transfer_time_ms(kv_bytes: int, bandwidth_gbps: float, latency_ms: float = 0.0) -> float:
    return (kv_bytes / (bandwidth_gbps * 1e9)) * 1000.0 + latency_ms


def analyze_kv_transfer(prompt_len: int, model_cfg: dict, hardware_cfg: dict) -> dict:
    kb = kv_cache_bytes(
        prompt_len,
        model_cfg["num_layers"],
        model_cfg["num_kv_heads"],
        model_cfg["head_dim"],
        model_cfg.get("dtype_bytes", 2),
    )
    p_ms = prefill_time_ms(prompt_len, model_cfg, hardware_cfg["prefill_tflops"])
    t_ms = transfer_time_ms(kb, hardware_cfg["bandwidth_gbps"], hardware_cfg.get("latency_ms", 0.0))
    return {
        "kv_bytes": kb,
        "prefill_ms": p_ms,
        "transfer_ms": t_ms,
        "ratio": t_ms / p_ms if p_ms > 0 else 0.0,
    }
