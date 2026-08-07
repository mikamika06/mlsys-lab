def classify_slices(events):
    res = {}
    for e in events:
        name = e.get("name", "")
        if "mm" in name or "kernel" in name:
            res[name] = "compute"
        else:
            res[name] = "overhead"
    return res
