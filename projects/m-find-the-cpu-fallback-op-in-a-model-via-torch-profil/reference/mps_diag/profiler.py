def find_fallback_ops(trace_events):
    fallbacks = []
    for ev in trace_events:
        if ev.get("cat") == "Memcpy" or (ev.get("cat") == "cpu_op" and "nonzero" in ev.get("name", "")):
            fallbacks.append(ev["name"])
    return sorted(list(set(fallbacks)))
