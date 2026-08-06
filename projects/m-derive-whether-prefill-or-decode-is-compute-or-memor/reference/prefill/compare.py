def compare_logs(cfg):
    """Compare TTFT and ITL from recorded logs."""
    entries = cfg["log_entries"]
    req_ttft = {}
    req_itl = {}
    for e in entries:
        rid = e["req_id"]
        if e["phase"] == "prefill":
            req_ttft[rid] = e["time_ms"]
        elif e["phase"] == "decode":
            req_itl.setdefault(rid, [])
            req_itl[rid].append(e["time_ms"])
    avg_ttft = sum(req_ttft.values()) / len(req_ttft) if req_ttft else 0.0
    all_itls = [t for lst in req_itl.values() for t in lst]
    avg_itl = sum(all_itls) / len(all_itls) if all_itls else 0.0
    return {"avg_ttft": float(avg_ttft), "avg_itl": float(avg_itl), "chunked": cfg["chunked"]}
