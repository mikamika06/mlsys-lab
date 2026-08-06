def choose_quant_under_budget(tensors, max_bytes, allowed_quants):
    best_q = allowed_quants[0]
    for q in allowed_quants:
        total = 0
        for t in tensors:
            bpp = 2 if q == "Q4_0" else 4
            if len(t.get("shape", [])) == 1:
                bpp = 4
            total += t["numel"] * bpp
        if total <= max_bytes:
            best_q = q
    return best_q
