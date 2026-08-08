def region_time_percentages(trace: dict) -> dict[str, float]:
    """Compute region percentage time breakdown from Proton trace."""
    events = trace.get("events", [])
    if not events:
        return {}

    total_dur = sum(e["dur"] for e in events)
    if total_dur == 0:
        return {}

    region_totals = {}
    for e in events:
        name = e["name"]
        region_totals[name] = region_totals.get(name, 0.0) + e["dur"]

    return {name: (dur / total_dur) * 100.0 for name, dur in region_totals.items()}
