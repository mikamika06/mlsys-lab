def format_table(profiles):
    lines = ["Mode | Size (Bytes) | Latency (ms) | Accuracy"]
    for p in profiles:
        lines.append(f"{p['mode']} | {p['size_bytes']} | {p['latency_ms']} | {p['accuracy']}")
    return "\n".join(lines)
