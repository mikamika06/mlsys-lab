def compute_pareto(configs):
    frontier = []
    for c in configs:
        dominated = False
        for other in configs:
            if other["id"] == c["id"]:
                continue
            better_or_equal_latency = other["latency"] <= c["latency"]
            better_or_equal_shmem = other["shmem"] <= c["shmem"]
            strictly_better = (other["latency"] < c["latency"]) or (other["shmem"] < c["shmem"])
            if better_or_equal_latency and better_or_equal_shmem and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(c)
    return sorted(frontier, key=lambda x: (x["latency"], x["shmem"]))
