import re


def parse_scrape(text: str) -> dict:
    """Parse raw Prometheus exposition text into typed metric families."""
    families = {}
    types = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split(maxsplit=3)
            if len(parts) >= 4 and parts[1] == "TYPE":
                types[parts[2]] = parts[3]
            continue
        match = re.match(
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([^\s]+)(?:\s+\d+)?$",
            line,
        )
        if not match:
            continue
        metric_name, raw_labels, val_str = match.groups()
        try:
            val = float(val_str)
        except ValueError:
            continue
        labels = {}
        if raw_labels:
            for label_pair in re.finditer(
                r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"([^"]*)"', raw_labels
            ):
                labels[label_pair.group(1)] = label_pair.group(2)
        family_name = metric_name
        for suffix in ("_bucket", "_count", "_sum", "_created"):
            if metric_name.endswith(suffix):
                base = metric_name[: -len(suffix)]
                if base in types:
                    family_name = base
                    break
        if family_name not in families:
            families[family_name] = {
                "name": family_name,
                "type": types.get(family_name, "untyped"),
                "samples": [],
            }
        families[family_name]["samples"].append(
            {"name": metric_name, "labels": labels, "value": val}
        )
    return families
