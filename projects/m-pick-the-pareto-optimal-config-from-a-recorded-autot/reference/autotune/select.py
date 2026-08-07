def select_best(configs, max_shmem):
    valid = [c for c in configs if c["shmem"] <= max_shmem]
    if not valid:
        return None
    best = min(valid, key=lambda x: (x["latency"], x["shmem"]))
    return best["id"]
