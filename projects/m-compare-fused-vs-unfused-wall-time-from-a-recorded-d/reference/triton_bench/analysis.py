def parse_benchmark(data):
    out = []
    for row in data:
        out.append({
            "name": row.get("name"),
            "size": int(row.get("size", 0)),
            "time_ms": float(row.get("time_ms", 0.0))
        })
    return out


def compute_ratios(fused_times, unfused_times):
    ratios = []
    for f, u in zip(fused_times, unfused_times):
        if f <= 0:
            ratios.append(0.0)
        else:
            ratios.append(float(u) / float(f))
    return ratios
