def analyze_memory_timeline(events: list) -> dict:
    curr = 0
    peak = 0
    peak_event_id = None
    category_totals = {}

    for ev in events:
        eid = ev.get("id")
        delta = ev.get("delta", 0)
        cat = ev.get("category", "unknown")
        curr += delta
        category_totals[cat] = category_totals.get(cat, 0) + delta
        if curr > peak:
            peak = curr
            peak_event_id = eid

    max_cat = max(category_totals, key=category_totals.get) if category_totals else None
    fix_map = {
        "hessian": "Use CPU offload for Hessian inverse calculation",
        "activations": "Enable activation disk/CPU offloading",
        "weights": "Process layer weights iteratively with immediate deletion",
        "workspace": "Reduce quantization block size parameter",
    }
    fix = fix_map.get(max_cat, "Enable CUDA memory defragmentation flags")

    return {
        "peak_bytes": peak,
        "peak_event_id": peak_event_id,
        "dominant_category": max_cat,
        "recommended_fix": fix,
    }
