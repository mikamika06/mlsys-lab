def aggregate_profile(data):
    total = 0.0
    records = data.get("records", [])
    for r in records:
        total += float(r.get("time_ms", 0.0))
    return {"total_time": total, "count": len(records)}
