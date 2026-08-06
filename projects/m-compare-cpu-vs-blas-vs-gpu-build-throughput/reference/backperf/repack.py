def analyze_size_invariance(fixture):
    before = fixture["size_before_bytes"]
    after = fixture["size_after_bytes"]
    delta = after - before
    unchanged = (delta == 0)
    return {
        "delta_bytes": delta,
        "unchanged": unchanged,
        "explanation": fixture["reason"]
    }
