def parse_truncated_log(log_text):
    lines = log_text.splitlines()
    events = []
    failed_at = None
    for line in lines:
        if "EVENT:" in line:
            parts = line.split("EVENT:")
            events.append(parts[1].strip())
        if "ERROR:" in line or "FATAL:" in line or "CRITICAL:" in line:
            failed_at = line.strip()
    return {"sequence": events, "failure_point": failed_at, "truncated": True}
