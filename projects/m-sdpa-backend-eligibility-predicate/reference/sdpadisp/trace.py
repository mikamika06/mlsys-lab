def detect_backend_from_trace(trace_events):
    for ev in trace_events:
        name = ev.get("name", "")
        if "flash_fwd" in name or "flash_attn" in name:
            return "flash_attention"
        if "mem_efficient" in name or "efficient_attention" in name:
            return "mem_efficient_attention"
        if "attn_naive" in name or "sdpa_math" in name or "math_kernel" in name:
            return "math"
    return "math"
