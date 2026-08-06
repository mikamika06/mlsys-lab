def parse_metrics(text: str) -> dict:
    metrics = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            name, val = parts[0], parts[1]
            try:
                metrics[name] = float(val)
            except ValueError:
                pass
    return metrics
