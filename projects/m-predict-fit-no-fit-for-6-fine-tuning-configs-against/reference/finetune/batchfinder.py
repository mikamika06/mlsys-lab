def find_largest_batch_size(curve_data, vram_budget_bytes):
    best_bs = 0
    for point in curve_data:
        bs = point["batch_size"]
        vram = point["vram_bytes"]
        if vram <= vram_budget_bytes:
            if bs > best_bs:
                best_bs = bs
    return best_bs
