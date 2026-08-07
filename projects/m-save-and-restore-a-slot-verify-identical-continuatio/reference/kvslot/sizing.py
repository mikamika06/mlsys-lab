def calculate_slot_sizing(vram_bytes, num_slots, context_len, hidden_size, num_layers):
    bytes_per_token = hidden_size * 4
    total_per_slot = context_len * bytes_per_token * num_layers
    max_slots = vram_bytes // total_per_slot if total_per_slot > 0 else 1
    return int(max(1, min(num_slots, max_slots)))
