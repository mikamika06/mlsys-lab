def parse_logs(logs):
    events = []
    for line in logs:
        parts = line.split(" ", 1)
        level = parts[0].strip("[]") if len(parts) > 1 else "INFO"
        msg = parts[1] if len(parts) > 1 else line
        events.append({"level": level, "message": msg})
    return events
