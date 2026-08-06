def apply_fix(latencies):
    return [l if l < 25.0 else 15.0 for l in latencies]
