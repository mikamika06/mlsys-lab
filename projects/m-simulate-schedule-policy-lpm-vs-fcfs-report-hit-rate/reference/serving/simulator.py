def simulate_schedule(requests: list[dict], policy: str) -> tuple[float, int]:
    cache = {}
    t = 0
    processed = set()
    total_tokens = sum(len(r["seq"]) for r in requests)
    total_hits = 0
    max_wait = 0

    while len(processed) < len(requests):
        avail = [r for r in requests if r["arrive"] <= t and r["id"] not in processed]
        if not avail:
            t = min(r["arrive"] for r in requests if r["id"] not in processed)
            continue

        if policy == "fcfs":
            avail.sort(key=lambda r: (r["arrive"], r["id"]))
            best = avail[0]
        elif policy == "lpm":
            def match_len(seq):
                for i in range(len(seq), 0, -1):
                    if seq[:i] in cache: return i
                return 0
            avail.sort(key=lambda r: (-match_len(r["seq"]), r["arrive"], r["id"]))
            best = avail[0]

        wait = t - best["arrive"]
        max_wait = max(max_wait, wait)

        seq = best["seq"]
        m_len = 0
        for i in range(len(seq), 0, -1):
            if seq[:i] in cache:
                m_len = i
                break
        total_hits += m_len

        for i in range(1, len(seq) + 1):
            cache[seq[:i]] = t

        processed.add(best["id"])
        t += 1

    return float(total_hits) / total_tokens, max_wait


def simulate_eviction(requests: list[dict], capacity: int, policy: str) -> float:
    cache = {}
    total_tokens = sum(len(r["seq"]) for r in requests)
    total_hits = 0
    t = 0

    for r in requests:
        seq = r["seq"]
        m_len = 0
        for i in range(len(seq), 0, -1):
            if seq[:i] in cache:
                m_len = i
                break
        total_hits += m_len

        for i in range(1, len(seq) + 1):
            p = seq[:i]
            if p not in cache:
                cache[p] = {"last_access": t, "freq": 0}
            cache[p]["last_access"] = t
            cache[p]["freq"] += 1

            while len(cache) > capacity:
                is_parent = set(x[:-1] for x in cache)
                leaves = [x for x in cache if x not in is_parent]

                if policy == "lru":
                    leaves.sort(key=lambda x: (cache[x]["last_access"], len(x), x))
                    del cache[leaves[0]]
                elif policy == "lfu":
                    leaves.sort(key=lambda x: (cache[x]["freq"], cache[x]["last_access"], len(x), x))
                    del cache[leaves[0]]
                elif policy == "lus":
                    leaves.sort(key=lambda x: (cache[x]["last_access"], len(x), x))
                    leaf = leaves[0]
                    target_time = cache[leaf]["last_access"]
                    highest = leaf
                    for j in range(len(leaf), 0, -1):
                        anc = leaf[:j]
                        if anc in cache and cache[anc]["last_access"] == target_time:
                            highest = anc
                    to_delete = [x for x in cache if x[:len(highest)] == highest]
                    for x in to_delete:
                        del cache[x]
        t += 1

    return float(total_hits) / total_tokens


def simulate_tiering(requests: list[dict], gpu_c: int, host_c: int) -> tuple[float, float]:
    gpu = {}
    host = {}
    gpu_hits = 0
    host_hits = 0
    total_tokens = sum(len(r["seq"]) for r in requests)
    t = 0

    for r in requests:
        seq = r["seq"]
        for i in range(1, len(seq) + 1):
            p = seq[:i]
            if p in gpu:
                gpu_hits += 1
            elif p in host:
                host_hits += 1
                del host[p]
                gpu[p] = t
            else:
                gpu[p] = t
            gpu[p] = t

            while len(gpu) > gpu_c:
                is_parent = set(x[:-1] for x in gpu)
                leaves = [x for x in gpu if x not in is_parent]
                leaves.sort(key=lambda x: (gpu[x], len(x), x))
                evict = leaves[0]
                host[evict] = gpu[evict]
                del gpu[evict]

            while len(host) > host_c:
                is_parent = set(x[:-1] for x in host)
                leaves = [x for x in host if x not in is_parent]
                leaves.sort(key=lambda x: (host[x], len(x), x))
                del host[leaves[0]]
        t += 1

    return float(gpu_hits) / total_tokens, float(host_hits) / total_tokens
