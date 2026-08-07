import numpy as np

def compute_slot_sizing(vram_bytes, num_slots, context_len, hidden_size, num_layers):
    bytes_per_token = hidden_size * 4
    total_per_slot = context_len * bytes_per_token * num_layers
    max_slots = vram_bytes // total_per_slot if total_per_slot > 0 else 1
    return int(max(1, min(num_slots, max_slots)))

def evaluate_unified_vs_perslot(num_chats, total_vram, slot_size):
    unified_vram = total_vram * 0.75
    perslot_vram = num_chats * slot_size
    efficient = perslot_vram <= total_vram
    return {"unified_vram": unified_vram, "perslot_vram": perslot_vram, "efficient": efficient}
