def detect_backend_from_trace(events: list) -> str:
    for ev in events:
        name = ev.get("name", "").lower()
        if "flash_attn" in name or "flashattention" in name:
            return "flash_attention"
        if "mem_efficient" in name or "efficient_attention" in name:
            return "mem_efficient"
        if "math" in name or "sdpa_kernel_math" in name:
            return "math"
    return "math"
