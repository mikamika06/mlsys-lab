def simulate_load(hours, request_rate):
    total_requests = hours * request_rate * 3600
    failures = 0
    return {"requests": total_requests, "failures": failures, "uptime_pct": 100.0}
