def build_profiles(records):
    out = []
    for r in records:
        out.append({
            "mode": r["mode"],
            "size_bytes": int(r["size_bytes"]),
            "latency_ms": float(r["latency_ms"]),
            "accuracy": float(r["accuracy"])
        })
    return out
