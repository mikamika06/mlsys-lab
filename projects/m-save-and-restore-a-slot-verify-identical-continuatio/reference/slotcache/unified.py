def compare_unified_per_slot(vram_bytes, num_chats, context_lens, bytes_per_token):
    total_requested = sum(context_lens) * bytes_per_token
    per_slot_static = max(context_lens) * bytes_per_token * num_chats
    unified_feasible = total_requested <= vram_bytes
    per_slot_feasible = per_slot_static <= vram_bytes
    return {
        "unified_bytes": total_requested,
        "per_slot_bytes": per_slot_static,
        "unified_feasible": unified_feasible,
        "per_slot_feasible": per_slot_feasible
    }
