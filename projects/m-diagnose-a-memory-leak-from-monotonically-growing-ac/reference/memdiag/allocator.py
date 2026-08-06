def compute_split_fraction(events):
    if not events:
        return 0.0
    splits = sum(1 for e in events if e.get("type") == "split")
    total = sum(1 for e in events if e.get("type") in ("split", "segment"))
    if total == 0:
        return 0.0
    return float(splits) / float(total)
