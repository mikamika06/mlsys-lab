def parse_trace_events(trace_data):
    """Extract PagedAdamW8bit events from a JSON trace dictionary or list."""
    raw_events = trace_data.get("traceEvents", trace_data if isinstance(trace_data, list) else [])
    parsed = []
    for ev in raw_events:
        if not isinstance(ev, dict):
            continue
        name = str(ev.get("name", ""))
        cat = str(ev.get("cat", ""))
        args = ev.get("args", {})
        if "PagedAdamW8bit" in name or "paged_adamw" in cat or args.get("is_paged", False):
            parsed.append({
                "name": name,
                "ts": float(ev.get("ts", 0.0)),
                "dur": float(ev.get("dur", 0.0)),
                "page_faults": int(args.get("page_faults", 0)),
                "bytes_transferred": int(args.get("bytes_transferred", 0)),
                "step": int(args.get("step", 0))
            })
    return parsed
