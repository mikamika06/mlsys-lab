def classify_snapshots(snapshots):
    """Classify six memory snapshots."""
    results = []
    for snap in snapshots:
        alloc = snap.get("allocated", 0)
        peak = snap.get("peak", 0)
        active = snap.get("active", 0)
        if alloc > 1000 and peak > 2000:
            results.append("leaking")
        elif active > 500:
            results.append("cached")
        else:
            results.append("stable")
    return results
