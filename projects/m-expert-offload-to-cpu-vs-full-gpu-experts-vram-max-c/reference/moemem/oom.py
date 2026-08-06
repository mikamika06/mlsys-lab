def diagnose_oom(model_config, flags):
    vram = model_config["vram_bytes"]
    base_req = model_config["base_bytes"]
    overhead = 0
    if flags.get("cpu_offload") and flags.get("kv_quant") and flags.get("mmap", True):
        overhead = int(base_req * 0.15)
    total_req = base_req + overhead
    if total_req > vram:
        return {"oom": True, "reason": "staging_overhead_exceeds_vram", "required": total_req, "available": vram}
    return {"oom": False, "reason": "safe", "required": total_req, "available": vram}
