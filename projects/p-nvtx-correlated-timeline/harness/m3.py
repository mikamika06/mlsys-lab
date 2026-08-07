import ref

def check(workdir):
    m = {"cost_identified": 0.0}
    try:
        from profiler.analysis import find_most_expensive_phase
        data = ref.get_sample_trace()
        res = find_most_expensive_phase(data)
        if res == "inference":
            m["cost_identified"] = 1.0
    except Exception:
        pass
    return m
