def calculate_slot_sizing(vram_bytes, context_len, bytes_per_token, overhead_bytes=1024*1024):
    available = vram_bytes - overhead_bytes
    if available <= 0:
        return {"max_np": 0, "max_context": 0}
    per_slot_cost = context_len * bytes_per_token
    max_np = max(1, available // per_slot_cost)
    max_context = available // bytes_per_token
    return {"max_np": int(max_np), "max_context": int(max_context)}
