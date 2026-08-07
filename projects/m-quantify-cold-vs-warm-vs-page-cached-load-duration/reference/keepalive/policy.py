def select_keep_alive(models, memory_cap_mb, request_frequencies):
    scored = []
    for m in models:
        mid = m["id"]
        freq = request_frequencies.get(mid, 0.0)
        score = freq * m["size_mb"]
        scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    current_mem = 0.0
    for _, m in scored:
        if current_mem + m["size_mb"] <= memory_cap_mb:
            selected.append(m["id"])
            current_mem += m["size_mb"]
    return selected
