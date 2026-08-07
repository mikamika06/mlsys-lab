def parse_exposition(text: str):
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line and "}" in line:
            name_part, rest = line.split("{", 1)
            labels_part, val_part = rest.split("}", 1)
            name = name_part.strip()
            val = float(val_part.strip())
            labels = {}
            for item in labels_part.split(","):
                if "=" in item:
                    k, v = item.split("=", 1)
                    labels[k.strip()] = v.strip().strip('"')
            metrics.setdefault(name, []).append((labels, val))
        else:
            parts = line.split()
            if len(parts) == 2:
                name, val = parts[0], float(parts[1])
                metrics.setdefault(name, []).append(({}, val))
    return metrics
