def compute_scores(config):
    res = {}
    for nid, node in config["nodes"].items():
        base = node["access_count"] * 2 + node["last_access"]
        if node["is_hot"]:
            base *= 10
        res[nid] = float(base)
    return res
