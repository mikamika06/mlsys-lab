def inspect_diff(engine_a, engine_b):
    diffs = []
    for k in engine_a:
        if engine_a[k] != engine_b.get(k):
            diffs.append(k)
    return diffs

def verify_tactics(engine_a, engine_b):
    return engine_a.get("tactics") == engine_b.get("tactics")
