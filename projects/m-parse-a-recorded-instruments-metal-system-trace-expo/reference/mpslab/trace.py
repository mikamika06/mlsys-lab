def parse_trace(text):
    counts = {}
    for line in text.strip().splitlines():
        if line.startswith("#") or line.startswith("Timestamp"):
            continue
        parts = line.split(",")
        if len(parts) >= 3:
            kind = parts[2].strip()
            counts[kind] = counts.get(kind, 0) + 1
    return counts
