def compute_primitive_dominance(verbose_logs):
    totals = {}
    total_time = 0.0
    
    for line in verbose_logs:
        line = line.strip()
        if not line.startswith("onednn,exec"):
            continue
        parts = line.split(",")
        if len(parts) < 7:
            continue
        prim_kind = parts[2]
        try:
            time_val = float(parts[-1])
        except ValueError:
            continue
            
        totals[prim_kind] = totals.get(prim_kind, 0.0) + time_val
        total_time += time_val

    breakdown = []
    if total_time > 0:
        for kind, t in sorted(totals.items(), key=lambda x: x[1], reverse=True):
            pct = round((t / total_time) * 100.0, 2)
            breakdown.append({
                'kind': kind,
                'time_ms': round(t, 4),
                'pct': pct
            })

    dominant_kind = breakdown[0]['kind'] if breakdown else ""

    return {
        'total_time_ms': round(total_time, 4),
        'breakdown': breakdown,
        'dominant_kind': dominant_kind
    }
