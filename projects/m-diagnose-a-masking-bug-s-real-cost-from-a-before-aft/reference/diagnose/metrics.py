def parse_metrics(raw_data):
    parsed = {}
    for line in raw_data.strip().split("\n"):
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            name, val = parts[0], float(parts[1])
            parsed[name] = val
    return parsed
