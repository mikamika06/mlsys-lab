from memacc.attention import calculate_eager_memory, calculate_sdpa_memory


def compare_attention_memory(configs):
    results = []
    for cfg in configs:
        b = cfg["batch_size"]
        s = cfg["seq_len"]
        h = cfg["num_heads"]
        d = cfg["head_dim"]
        dt = cfg.get("dtype_str", "float16")

        eager_res = calculate_eager_memory(b, s, h, d, dt)
        sdpa_res = calculate_sdpa_memory(b, s, h, d, dt)

        saving_bytes = eager_res["stored_forward_bytes"] - sdpa_res["stored_forward_bytes"]
        ratio = (
            sdpa_res["stored_forward_bytes"] / eager_res["stored_forward_bytes"]
            if eager_res["stored_forward_bytes"] > 0
            else 1.0
        )

        results.append({
            "batch_size": b,
            "seq_len": s,
            "num_heads": h,
            "head_dim": d,
            "eager_stored_bytes": eager_res["stored_forward_bytes"],
            "sdpa_stored_bytes": sdpa_res["stored_forward_bytes"],
            "eager_bwd_peak_bytes": eager_res["bwd_peak_bytes"],
            "sdpa_bwd_peak_bytes": sdpa_res["bwd_peak_bytes"],
            "memory_saved_bytes": saving_bytes,
            "sdpa_to_eager_ratio": ratio,
        })
    return results


def find_max_sequence_length(model_cfg, vram_budget_bytes, mode="sdpa"):
    b = model_cfg["batch_size"]
    h = model_cfg["num_heads"]
    d = model_cfg["head_dim"]
    dt = model_cfg.get("dtype_str", "float16")

    fn = calculate_sdpa_memory if mode == "sdpa" else calculate_eager_memory

    low = 1
    high = 1048576
    best = 0

    while low <= high:
        mid = (low + high) // 2
        res = fn(b, mid, h, d, dt)
        if res["bwd_peak_bytes"] <= vram_budget_bytes:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best
