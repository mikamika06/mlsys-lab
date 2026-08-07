def detect_backend_from_trace(trace_events):
    for event in trace_events:
        name = event.get("name", "")
        if "flash" in name.lower() or "fmha" in name.lower():
            return "flash_attention"
        if "efficient" in name.lower() or "mem_eff" in name.lower():
            return "efficient_attention"
        if "math" in name.lower() or "sdpa_kernel_math" in name.lower():
            return "math"
    return "math"
