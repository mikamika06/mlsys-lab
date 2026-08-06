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


def simulate_pipeline(requests: list[dict], num_p: int, num_d: int, model_cfg: dict, hardware_cfg: dict) -> dict:
    if not requests:
        return {
            "avg_ttft_ms": 0.0,
            "avg_tpot_ms": 0.0,
            "total_makespan_ms": 0.0,
            "p_utilization": 0.0,
            "d_utilization": 0.0,
        }
    p_free = [0.0] * num_p
    d_free = [0.0] * num_d

    total_ttft = 0.0
    total_tpot = 0.0
    p_busy = 0.0
    d_busy = 0.0
    max_finish = 0.0

    sorted_reqs = sorted(requests, key=lambda x: x["arrival_ms"])

    for req in sorted_reqs:
        arr = float(req["arrival_ms"])
        plen = int(req["prompt_len"])
        gtok = int(req["gen_tokens"])

        kb = kv_cache_bytes(
            plen,
            model_cfg["num_layers"],
            model_cfg["num_kv_heads"],
            model_cfg["head_dim"],
            model_cfg.get("dtype_bytes", 2),
        )
        p_ms = prefill_time_ms(plen, model_cfg, hardware_cfg["prefill_tflops"])
        trans_ms = transfer_time_ms(kb, hardware_cfg["bandwidth_gbps"], hardware_cfg.get("latency_ms", 0.0))
        p_total = p_ms + trans_ms

        p_idx = min(range(num_p), key=lambda i: p_free[i])
        p_start = max(arr, p_free[p_idx])
        p_end = p_start + p_total
        p_free[p_idx] = p_end
        p_busy += p_total

        ttft = p_end - arr
        total_ttft += ttft

        dec_step_ms = decode_step_time_ms(plen, model_cfg, hardware_cfg["decode_tflops"])
        d_total = dec_step_ms * gtok

        d_idx = min(range(num_d), key=lambda i: d_free[i])
        d_start = max(p_end, d_free[d_idx])
        d_end = d_start + d_total
        d_free[d_idx] = d_end
        d_busy += d_total

        tpot = dec_step_ms if gtok > 0 else 0.0
        total_tpot += tpot

        if d_end > max_finish:
            max_finish = d_end

    n = len(sorted_reqs)
    avg_ttft = total_ttft / n
    avg_tpot = total_tpot / n
    p_util = p_busy / (num_p * max_finish) if max_finish > 0 else 0.0
    d_util = d_busy / (num_d * max_finish) if max_finish > 0 else 0.0

    return {
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
        "total_makespan_ms": max_finish,
        "p_utilization": p_util,
        "d_utilization": d_util,
    }


MODEL_7B = {
    "num_layers": 32,
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "hidden_dim": 4096,
    "intermediate_dim": 11008,
    "dtype_bytes": 2,
}

MODEL_70B = {
    "num_layers": 80,
    "num_heads": 64,
    "num_kv_heads": 8,
    "head_dim": 128,
    "hidden_dim": 8192,
    "intermediate_dim": 28672,
    "dtype_bytes": 2,
}

HW_FAST_NET = {
    "prefill_tflops": 300.0,
    "decode_tflops": 300.0,
    "bandwidth_gbps": 100.0,
    "latency_ms": 0.5,
}

HW_SLOW_NET = {
    "prefill_tflops": 300.0,
    "decode_tflops": 300.0,
    "bandwidth_gbps": 25.0,
    "latency_ms": 2.0,
}

TEST_CONFIGS = [
    (32768, MODEL_7B, HW_FAST_NET),
    (32768, MODEL_7B, HW_SLOW_NET),
    (32768, MODEL_70B, HW_FAST_NET),
    (4096, MODEL_7B, HW_FAST_NET),
    (16384, MODEL_70B, HW_SLOW_NET),
]

SIZING_CONFIGS = [
    (150.0, 50.0, 1.2, 100),
    (300.0, 100.0, 2.5, 200),
    (80.0, 0.0, 0.8, 50),
    (500.0, 250.0, 1.5, 300),
]

ALLOC_CONFIGS = [
    (8, 200.0, 240.0),
    (16, 400.0, 100.0),
    (12, 150.0, 150.0),
]

SIM_CONFIGS = [
    (
        [
            {"arrival_ms": 0.0, "prompt_len": 32768, "gen_tokens": 128},
            {"arrival_ms": 10.0, "prompt_len": 32768, "gen_tokens": 128},
            {"arrival_ms": 20.0, "prompt_len": 32768, "gen_tokens": 64},
            {"arrival_ms": 30.0, "prompt_len": 16384, "gen_tokens": 256},
        ],
        2,
        2,
        MODEL_7B,
        HW_FAST_NET,
    ),
    (
        [
            {"arrival_ms": 0.0, "prompt_len": 32768, "gen_tokens": 64},
            {"arrival_ms": 5.0, "prompt_len": 32768, "gen_tokens": 64},
        ],
        1,
        2,
        MODEL_7B,
        HW_SLOW_NET,
    ),
]
