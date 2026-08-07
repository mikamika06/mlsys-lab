def compute_flops(events):
    """Compute achieved FLOP/s per matmul from with_flops."""
    results = {}
    for ev in events:
        if ev.get("cat") == "cpu_op":
            flops = ev.get("args", {}).get("with_flops", 0)
            dur = ev.get("dur", 1)
            if dur > 0:
                achieved = flops / (dur * 1e-6)
                results[ev["name"]] = float(achieved)
    return results
