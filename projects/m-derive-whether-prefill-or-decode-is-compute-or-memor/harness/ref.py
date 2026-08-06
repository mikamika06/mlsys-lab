def get_roofline_cases():
    return [
        {
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "vocab_size": 32000,
            "batch_size": 1,
            "seq_len": 2048,
            "is_prefill": True,
            "peak_tflops": 312.0,
            "mem_bw_gbs": 1555.0
        },
        {
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "vocab_size": 32000,
            "batch_size": 16,
            "seq_len": 1,
            "is_prefill": False,
            "peak_tflops": 312.0,
            "mem_bw_gbs": 1555.0
        },
        {
            "hidden_size": 8192,
            "num_layers": 80,
            "num_heads": 64,
            "num_kv_heads": 8,
            "vocab_size": 32000,
            "batch_size": 4,
            "seq_len": 512,
            "is_prefill": True,
            "peak_tflops": 989.0,
            "mem_bw_gbs": 3350.0
        }
    ]

def get_simulation_cases():
    return [
        {
            "token_budget": 2048,
            "decodes": [{"id": 1, "tokens_gen": 10}, {"id": 2, "tokens_gen": 5}],
            "prefills": [{"id": 101, "remaining_tokens": 1024, "max_chunk": 512}, {"id": 102, "remaining_tokens": 4000, "max_chunk": 512}]
        },
        {
            "token_budget": 512,
            "decodes": [{"id": 3, "tokens_gen": 2}, {"id": 4, "tokens_gen": 8}, {"id": 5, "tokens_gen": 1}],
            "prefills": [{"id": 103, "remaining_tokens": 200, "max_chunk": 256}]
        }
    ]

def get_log_comparison_cases():
    return [
        {
            "log_entries": [
                {"req_id": 1, "phase": "prefill", "tokens": 512, "time_ms": 20.0},
                {"req_id": 1, "phase": "decode", "tokens": 1, "time_ms": 5.0},
                {"req_id": 1, "phase": "decode", "tokens": 1, "time_ms": 5.2},
                {"req_id": 2, "phase": "prefill", "tokens": 2048, "time_ms": 80.0},
                {"req_id": 2, "phase": "decode", "tokens": 1, "time_ms": 6.0}
            ],
            "chunked": True
        },
        {
            "log_entries": [
                {"req_id": 1, "phase": "prefill", "tokens": 512, "time_ms": 45.0},
                {"req_id": 1, "phase": "decode", "tokens": 1, "time_ms": 4.8},
                {"req_id": 2, "phase": "prefill", "tokens": 2048, "time_ms": 180.0},
                {"req_id": 2, "phase": "decode", "tokens": 1, "time_ms": 5.5}
            ],
            "chunked": False
        }
    ]

def analyze_roofline(cfg):
    h = cfg["hidden_size"]
    l = cfg["num_layers"]
    b = cfg["batch_size"]
    s = cfg["seq_len"]
    flops = b * s * l * (12.0 * h * h + 2.0 * h * s)
    param_bytes = l * 12.0 * h * h * 2.0
    kv_bytes = 2.0 * b * s * l * (cfg["num_kv_heads"] * (h // cfg["num_heads"]) * 2.0)
    total_bytes = param_bytes + kv_bytes
    intensity = flops / total_bytes if total_bytes > 0 else 0.0
    ridge = (cfg["peak_tflops"] * 1e12) / (cfg["mem_bw_gbs"] * 1e9)
    bound = "compute" if intensity >= ridge else "memory"
    return {"arithmetic_intensity": float(intensity), "bound": bound, "estimated_flops": float(flops), "total_bytes": float(total_bytes)}

def simulate_batch(cfg):
    budget = cfg["token_budget"]
    decodes = cfg["decodes"]
    prefills = cfg["prefills"]
    dec_tokens_used = len(decodes)
    remaining_budget = budget - dec_tokens_used
    if remaining_budget < 0:
        selected_decodes = decodes[:budget]
        selected_prefills = []
        return {"decodes": selected_decodes, "prefills": selected_prefills, "tokens_used": budget}
    selected_decodes = list(decodes)
    selected_prefills = []
    for p in prefills:
        if remaining_budget <= 0:
            break
        alloc = min(p["remaining_tokens"], p["max_chunk"], remaining_budget)
        if alloc > 0:
            selected_prefills.append({"id": p["id"], "allocated_tokens": alloc})
            remaining_budget -= alloc
    tokens_used = budget - remaining_budget
    return {"decodes": selected_decodes, "prefills": selected_prefills, "tokens_used": tokens_used}

def compare_logs(cfg):
    entries = cfg["log_entries"]
    req_ttft = {}
    req_itl = {}
    for e in entries:
        rid = e["req_id"]
        if e["phase"] == "prefill":
            req_ttft[rid] = e["time_ms"]
        elif e["phase"] == "decode":
            req_itl.setdefault(rid, [])
            req_itl[rid].append(e["time_ms"])
    avg_ttft = sum(req_ttft.values()) / len(req_ttft) if req_ttft else 0.0
    all_itls = [t for lst in req_itl.values() for t in lst]
    avg_itl = sum(all_itls) / len(all_itls) if all_itls else 0.0
    return {"avg_ttft": float(avg_ttft), "avg_itl": float(avg_itl), "chunked": cfg["chunked"]}
