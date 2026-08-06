def parse_logs(logs):
    parsed = {}
    for l in logs:
        run_id = l["run_id"]
        lats = l["latencies"]
        parsed[run_id] = sum(lats) / len(lats)
    return parsed
