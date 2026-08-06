PROFILES = [
    {"name": "low-vram", "vram_bytes": 6 * 1024**3, "model_bytes": 4 * 1024**3},
    {"name": "mid-vram", "vram_bytes": 16 * 1024**3, "model_bytes": 6 * 1024**3},
    {"name": "high-vram", "vram_bytes": 24 * 1024**3, "model_bytes": 8 * 1024**3},
]


def default_context(profile):
    if profile["vram_bytes"] <= 8 * 1024**3:
        return 4096
    return 32768


def compute_slots(num_ctx, parallel):
    return [{"slot_id": i, "ctx_len": num_ctx // parallel} for i in range(parallel)]


def predict_cliff(model_bytes, kv_bytes_per_token, ctx_len, parallel, vram_bytes):
    total_kv = kv_bytes_per_token * ctx_len * parallel
    resident = model_bytes + total_kv
    return {
        "resident_bytes": resident,
        "fits": resident <= vram_bytes,
        "headroom": vram_bytes - resident,
    }
