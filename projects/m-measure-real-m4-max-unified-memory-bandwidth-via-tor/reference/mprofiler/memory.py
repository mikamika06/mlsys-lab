def build_memory_timeline(alloc_samples, driver_samples):
    timeline = []
    for a, d in zip(alloc_samples, driver_samples):
        timeline.append({"allocated": int(a), "driver": int(d), "diff": int(d - a)})
    return timeline
