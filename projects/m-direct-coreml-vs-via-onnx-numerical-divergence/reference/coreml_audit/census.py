def node_census(nodes):
    counts = {}
    for node in nodes:
        key = (node["op"], node["provider"])
        counts[key] = counts.get(key, 0) + 1
    return sorted([{"op": k[0], "provider": k[1], "count": v} for k, v in counts.items()], key=lambda x: (x["op"], x["provider"]))
