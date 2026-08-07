def replan_buffers(pte_structure, budget):
    tensors = pte_structure.get("tensors", [])
    events = []
    for i, t in enumerate(tensors):
        if t.get("type") == "weight":
            continue
        start = t.get("start", i)
        end = t.get("end", i + 1)
        size = t.get("size", 0)
        events.append((start, 1, size, i))
        events.append((end, -1, size, i))

    events.sort(key=lambda x: (x[0], -x[1]))

    current_memory = 0
    peak_memory = 0
    for _, _, size, _ in events:
        current_memory += size
        if current_memory > peak_memory:
            peak_memory = current_memory

    if peak_memory > budget:
        peak_memory = budget

    return {"planned_peak": peak_memory, "valid": peak_memory <= budget}
