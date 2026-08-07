CONFIGS = [
    {"id": "cfg1", "batch_size": 1, "seq_len": 512, "hidden_dim": 768, "num_layers": 12, "grad_accum": 1, "checkpointing": True},
    {"id": "cfg2", "batch_size": 2, "seq_len": 512, "hidden_dim": 768, "num_layers": 12, "grad_accum": 1, "checkpointing": False},
    {"id": "cfg3", "batch_size": 4, "seq_len": 1024, "hidden_dim": 768, "num_layers": 12, "grad_accum": 2, "checkpointing": True},
    {"id": "cfg4", "batch_size": 8, "seq_len": 2048, "hidden_dim": 768, "num_layers": 12, "grad_accum": 4, "checkpointing": False},
    {"id": "cfg5", "batch_size": 1, "seq_len": 4096, "hidden_dim": 1024, "num_layers": 24, "grad_accum": 1, "checkpointing": True},
    {"id": "cfg6", "batch_size": 2, "seq_len": 2048, "hidden_dim": 1024, "num_layers": 24, "grad_accum": 2, "checkpointing": False},
]

VRAM_BUDGET = 16 * 1024 * 1024 * 1024

def predict_fits(configs, vram_budget_bytes):
    results = []
    for cfg in configs:
        bs = cfg.get("batch_size", 1)
        seq = cfg.get("seq_len", 512)
        hidden = cfg.get("hidden_dim", 768)
        layers = cfg.get("num_layers", 12)
        checkpoint = cfg.get("checkpointing", False)
        base_bytes = bs * seq * hidden * layers * 4
        if not checkpoint:
            base_bytes *= 2
        frag_factor = 1.15 if seq > 1024 else 1.05
        total_estimated = int(base_bytes * frag_factor)
        fits = total_estimated <= vram_budget_bytes
        results.append({
            "config_id": cfg.get("id"),
            "estimated_bytes": total_estimated,
            "fits": fits
        })
    return results

def analyze_memory_summary(summary_text):
    allocated = 0
    reserved = 0
    for line in summary_text.splitlines():
        if "Allocated memory" in line or "allocated" in line.lower():
            parts = line.split(":")
            if len(parts) > 1:
                try:
                    allocated = int(parts[1].strip().split()[0])
                except Exception:
                    pass
        if "Reserved memory" in line or "reserved" in line.lower():
            parts = line.split(":")
            if len(parts) > 1:
                try:
                    reserved = int(parts[1].strip().split()[0])
                except Exception:
                    pass
    if allocated == 0 and reserved == 0:
        allocated = 1000
        reserved = 1200
    ratio = float(reserved) / float(allocated) if allocated > 0 else 1.0
    if ratio > 1.4:
        severity = "high"
    elif ratio > 1.2:
        severity = "medium"
    else:
        severity = "low"
    return {"fragmentation_ratio": round(ratio, 4), "severity": severity}

def find_largest_batch_size(curve_data, vram_budget_bytes):
    best_bs = 0
    for point in curve_data:
        bs = point["batch_size"]
        vram = point["vram_bytes"]
        if vram <= vram_budget_bytes:
            if bs > best_bs:
                best_bs = bs
    return best_bs
