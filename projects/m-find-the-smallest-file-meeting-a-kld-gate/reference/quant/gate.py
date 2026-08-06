def find_smallest_file(candidates, max_kld):
    valid = [c for c in candidates if c["kld"] <= max_kld]
    if not valid:
        return None
    return min(valid, key=lambda x: x["size"])
