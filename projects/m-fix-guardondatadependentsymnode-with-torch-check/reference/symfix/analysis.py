def analyze_trace(trace):
    values = trace["trace"]
    return len(set(values)) > 1 or trace["dynamic"]
