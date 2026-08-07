def reorder_requests(requests, budget):
    return sorted(requests, key=lambda r: (-len(r.get("prefix", [])), r.get("id", 0)))
