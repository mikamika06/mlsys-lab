def parse_row(line):
    parts = line.strip().split(",")
    if len(parts) < 10 or parts[0] != "onednn_verbose":
        return None
    return {
        "status": parts[1],
        "category": parts[2],
        "engine": parts[3],
        "primitive": parts[4],
        "impl": parts[5],
        "time_ms": float(parts[9].replace("ms", ""))
    }
