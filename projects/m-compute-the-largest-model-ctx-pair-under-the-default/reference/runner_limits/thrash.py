def detect_swap_thrash(stream, tok_s_thresh=5.0, pressure_thresh=0.85):
    for entry in stream:
        tok_s = entry.get("tok_s", 10.0)
        pressure = entry.get("memory_pressure", 0.0)
        swapped = entry.get("swap_active", False)
        if tok_s < tok_s_thresh and (pressure > pressure_thresh or swapped):
            return True
    return False
