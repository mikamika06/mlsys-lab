def compare_unified_vs_perslot(num_chats, total_vram, slot_size):
    unified_vram = total_vram * 0.75
    perslot_vram = num_chats * slot_size
    efficient = perslot_vram <= total_vram
    return {"unified_vram": unified_vram, "perslot_vram": perslot_vram, "efficient": efficient}
