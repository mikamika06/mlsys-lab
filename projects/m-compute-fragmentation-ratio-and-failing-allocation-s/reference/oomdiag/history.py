def find_largest_allocation_site(snapshot):
    best_site = None
    max_bytes = -1
    for item in snapshot:
        if item["bytes"] > max_bytes:
            max_bytes = item["bytes"]
            best_site = item["site"]
    return best_site
