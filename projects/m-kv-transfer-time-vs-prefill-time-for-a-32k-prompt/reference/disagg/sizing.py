def decode_step_flops(prompt_len: int, model_cfg: dict) -> int:
    l = model_cfg["num_layers"]
    h = model_cfg["hidden_dim"]
    hq = model_cfg["num_heads"] * model_cfg["head_dim"]
    hkv = model_cfg["num_kv_heads"] * model_cfg["head_dim"]
    d_int = model_cfg["intermediate_dim"]
    gemm_flops = 2 * h * (2 * hq + 2 * hkv) + 6 * h * d_int
    attn_flops = 2 * model_cfg["num_heads"] * model_cfg["head_dim"] * prompt_len
    return l * (gemm_flops + attn_flops)


def decode_step_time_ms(prompt_len: int, model_cfg: dict, tflops: float) -> float:
    flops = decode_step_flops(prompt_len, model_cfg)
    return (flops / (tflops * 1e12)) * 1000.0


def compute_pd_ratio(prefill_ms: float, transfer_ms: float, decode_step_ms: float, gen_tokens: int) -> float:
    t_prefill_total = prefill_ms + transfer_ms
    t_decode_total = decode_step_ms * gen_tokens
    if t_decode_total == 0:
        return 0.0
    return t_prefill_total / t_decode_total


def allocate_pd_nodes(total_nodes: int, prefill_total_ms: float, decode_total_ms: float) -> tuple[int, int]:
    best_np = 1
    best_nd = total_nodes - 1
    best_rate = -1.0
    min_diff = float("inf")
    for np in range(1, total_nodes):
        nd = total_nodes - np
        rate_p = np / prefill_total_ms if prefill_total_ms > 0 else float("inf")
        rate_d = nd / decode_total_ms if decode_total_ms > 0 else float("inf")
        min_rate = min(rate_p, rate_d)
        diff = abs(rate_p - rate_d)
        if min_rate > best_rate + 1e-9:
            best_rate = min_rate
            min_diff = diff
            best_np, best_nd = np, nd
        elif abs(min_rate - best_rate) <= 1e-9 and diff < min_diff:
            min_diff = diff
            best_np, best_nd = np, nd
    return best_np, best_nd
