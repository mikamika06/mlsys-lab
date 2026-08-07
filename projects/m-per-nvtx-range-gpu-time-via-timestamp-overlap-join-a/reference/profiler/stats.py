def aggregate_cupti(kernel_rows):
    stats = {}
    for k in kernel_rows:
        name = k["name"]
        dur = k["end"] - k["start"]
        if name not in stats:
            stats[name] = {"count": 0, "total": 0.0, "min": float("inf"), "max": float("-inf")}
        s = stats[name]
        s["count"] += 1
        s["total"] += dur
        if dur < s["min"]:
            s["min"] = dur
        if dur > s["max"]:
            s["max"] = dur

    out = {}
    for name, s in stats.items():
        cnt = s["count"]
        tot = s["total"]
        avg = tot / cnt if cnt > 0 else 0.0
        out[name] = {
            "count": cnt,
            "total": tot,
            "avg": avg,
            "min": s["min"],
            "max": s["max"]
        }
    return out
