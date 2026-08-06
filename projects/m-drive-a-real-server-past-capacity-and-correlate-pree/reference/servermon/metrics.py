def parse_trace(raw_trace):
    return {
        "load_factor": float(raw_trace["load_factor"]),
        "preemptions": [int(x) for x in raw_trace["preemptions"]],
        "latencies": [float(x) for x in raw_trace["latencies"]]
    }
