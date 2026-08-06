MODELS = [
    {"name": "model-7b", "params": 7e9, "bytes_per_param": 2.0, "kv_per_token": 0.5e6},
    {"name": "model-13b", "params": 13e9, "bytes_per_param": 2.0, "kv_per_token": 1.0e6},
    {"name": "model-70b", "params": 70e9, "bytes_per_param": 2.0, "kv_per_token": 2.0e6},
]
DEFAULT_CEILING = 16 * 1024 * 1024 * 1024

def compute_largest_pair(models, ceiling, overhead=1024*1024*1024):
    best_model = None
    best_ctx = 0
    for m in models:
        base_mem = m["params"] * m["bytes_per_param"] + overhead
        if base_mem >= ceiling:
            continue
        available = ceiling - base_mem
        max_ctx = int(available // m["kv_per_token"])
        if max_ctx > 0:
            if best_model is None or m["params"] > best_model["params"] or (m["params"] == best_model["params"] and max_ctx > best_ctx):
                best_model = m["name"]
                best_ctx = max_ctx
    return best_model, best_ctx

def detect_swap_thrash(stream, tok_s_thresh=5.0, pressure_thresh=0.85):
    for entry in stream:
        tok_s = entry.get("tok_s", 10.0)
        pressure = entry.get("memory_pressure", 0.0)
        swapped = entry.get("swap_active", False)
        if tok_s < tok_s_thresh and (pressure > pressure_thresh or swapped):
            return True
    return False

def diagnose_log(log_text):
    if "Metal buffer allocation failed" in log_text or "IOReturn(-12)" in log_text:
        return "metal_alloc_failure"
    if "Killed: 9" in log_text or "out of memory" in log_text.lower() or "oom" in log_text.lower():
        return "oom_kill"
    if "segmentation fault" in log_text.lower() or "core dumped" in log_text.lower() or "abort" in log_text.lower():
        return "runner_crash"
    return "success"
