def parse_verbose_log(log_content: str) -> list:
    events = []
    for line in log_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "reorder" in line:
            parts = line.split(",")
            events.append({
                "primitive": "reorder",
                "src_format": parts[1] if len(parts) > 1 else "unknown",
                "dst_format": parts[2] if len(parts) > 2 else "unknown"
            })
    return events
