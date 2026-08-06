def classify_snapshots(snapshots):
    results = []
    for snap in snapshots:
        allocated = 0
        if isinstance(snap, dict) and "segments" in snap:
            for seg in snap["segments"]:
                allocated += seg.get("allocated_size", seg.get("size", 0))
        elif isinstance(snap, list):
            for item in snap:
                if isinstance(item, dict):
                    allocated += item.get("allocated_size", item.get("size", 0))
        elif isinstance(snap, (int, float)):
            allocated = int(snap)
        results.append(allocated)
    if len(results) < 2:
        return "stable"
    diffs = [results[i+1] - results[i] for i in range(len(results)-1)]
    avg_diff = sum(diffs) / len(diffs)
    if avg_diff > 100:
        return "leaking"
    elif avg_diff > 0:
        return "growing"
    else:
        return "stable"
