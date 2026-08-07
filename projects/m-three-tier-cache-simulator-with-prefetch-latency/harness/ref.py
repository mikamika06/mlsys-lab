def simulate(trace, l1_cap, l2_cap, policy="always", write_mode="wb"):
    freq_map = {}
    l1 = {}
    l2 = {}
    l1_used = 0
    l2_used = 0

    latency_ns = 0
    write_penalty_ns = 0
    l1_evicts = 0
    l2_evicts = 0

    for op, key, size in trace:
        freq_map[key] = freq_map.get(key, 0) + 1
        admit_l1 = False
        if policy == "always":
            admit_l1 = True
        elif policy == "reuse_2":
            admit_l1 = freq_map[key] >= 2
        elif policy == "size_aware":
            admit_l1 = size <= (l1_cap // 4)

        if key in l1:
            latency_ns += size * 1
            node = l1.pop(key)
            l1[key] = node
            if op == "W":
                if write_mode == "wb":
                    node["dirty"] = True
                else:
                    write_penalty_ns += size * 110
        elif key in l2:
            latency_ns += size * 10
            node = l2.pop(key)
            l2_used -= node["size"]
            if admit_l1:
                if op == "W":
                    if write_mode == "wb":
                        node["dirty"] = True
                    else:
                        write_penalty_ns += size * 110
                l1[key] = node
                l1_used += node["size"]
            else:
                if op == "W":
                    if write_mode == "wb":
                        node["dirty"] = True
                    else:
                        write_penalty_ns += size * 100
                l2[key] = node
                l2_used += node["size"]
        else:
            latency_ns += size * 100
            node = {"size": size, "dirty": False}
            if admit_l1:
                if op == "W":
                    if write_mode == "wb":
                        node["dirty"] = True
                    else:
                        write_penalty_ns += size * 110
                l1[key] = node
                l1_used += size
            else:
                if op == "W":
                    if write_mode == "wb":
                        node["dirty"] = True
                    else:
                        write_penalty_ns += size * 100
                l2[key] = node
                l2_used += size

        while l1_used > l1_cap:
            k, v = next(iter(l1.items()))
            del l1[k]
            l1_used -= v["size"]
            l1_evicts += 1
            if v["dirty"]:
                write_penalty_ns += v["size"] * 10
            l2[k] = v
            l2_used += v["size"]

        while l2_used > l2_cap:
            k, v = next(iter(l2.items()))
            del l2[k]
            l2_used -= v["size"]
            l2_evicts += 1
            if v["dirty"]:
                write_penalty_ns += v["size"] * 100

    return {
        "latency_ns": latency_ns,
        "write_penalty_ns": write_penalty_ns,
        "l1_evictions": l1_evicts,
        "l2_evictions": l2_evicts
    }

TRACES = [
    [("R", 1, 1024), ("W", 2, 512), ("R", 1, 1024), ("W", 3, 2048), ("R", 2, 512)],
    [("R", i % 5, 256) for i in range(20)] + [("W", i % 3, 1024) for i in range(5)],
    [("W", 1, 4096), ("R", 1, 4096), ("R", 2, 512), ("W", 2, 512), ("R", 1, 4096)]
]
