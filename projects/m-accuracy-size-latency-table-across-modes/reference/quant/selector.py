def select_mode(profiles, max_size_ratio, min_accuracy):
    valid = []
    for p in profiles:
        if p["size_ratio"] <= max_size_ratio and p["accuracy"] >= min_accuracy:
            valid.append(p)
    if not valid:
        return None
    valid.sort(key=lambda x: (x["latency"], -x["accuracy"]))
    return valid[0]["mode"]
