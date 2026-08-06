def routing_fractions(parsed_ops):
    """Compute device routing fractions across ANE, GPU, and CPU."""
    if not parsed_ops:
        return {"ANE": 0.0, "GPU": 0.0, "CPU": 0.0}
    counts = {"ANE": 0, "GPU": 0, "CPU": 0}
    for op in parsed_ops:
        dev = op.get("device", "CPU")
        if dev in counts:
            counts[dev] += 1
        else:
            counts["CPU"] += 1
    total = len(parsed_ops)
    return {dev: counts[dev] / total for dev in ["ANE", "GPU", "CPU"]}
