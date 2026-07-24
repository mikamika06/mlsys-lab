def grade(sol, fx) -> dict:
    line_bytes = 64

    # Define the 5 layouts: each is a function t -> byte_address
    def layout_addrs(layout_id, num_threads=4):
        t_range = list(range(num_threads))
        if layout_id == 0:
            return [t * 8 for t in t_range]
        elif layout_id == 1:
            return [t * 64 for t in t_range]
        elif layout_id == 2:
            return [t * 128 for t in t_range]
        elif layout_id == 3:
            return [t * 8 + 64 * (t % 2) for t in t_range]
        elif layout_id == 4:
            return [t * 16 for t in t_range]

    def has_false_sharing(addrs, lb):
        lines = [a // lb for a in addrs]
        return len(lines) != len(set(lines))

    # Compute reference labels
    ref = [has_false_sharing(layout_addrs(i), line_bytes) for i in range(5)]

    try:
        result = list(sol.classify_layouts(line_bytes))
    except Exception:
        return {"exact_match": 0.0}

    if len(result) != 5:
        return {"exact_match": 0.0}

    matches = sum(1 for r, e in zip(result, ref) if bool(r) == bool(e))
    exact_match = 1.0 if matches == 5 else 0.0
    return {"exact_match": exact_match}
