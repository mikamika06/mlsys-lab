def compute_counter_rates(
    family1: dict, family2: dict, duration_seconds: float
) -> dict:
    """Compute per-second throughput rates with counter reset handling."""
    results = {}
    if duration_seconds <= 0:
        return results
    s1_map = {
        frozenset(s["labels"].items()): s["value"]
        for s in family1.get("samples", [])
    }
    s2_map = {
        frozenset(s["labels"].items()): s["value"]
        for s in family2.get("samples", [])
    }
    for key, v2 in s2_map.items():
        if key in s1_map:
            v1 = s1_map[key]
            if v2 >= v1:
                delta = v2 - v1
            else:
                delta = v2
            results[key] = delta / duration_seconds
    return results
