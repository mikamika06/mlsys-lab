from radixtree.tree import TokenRadixTree

def schedule_requests(requests, cache_tree, policy="lpm"):
    if policy == "fcfs":
        return list(requests)
    elif policy == "lpm":
        reqs = list(requests)
        scheduled = []
        while reqs:
            best_idx = 0
            best_match_len = -1
            for i, req in enumerate(reqs):
                matched, _ = cache_tree.longest_match(req)
                if len(matched) > best_match_len:
                    best_match_len = len(matched)
                    best_idx = i
            chosen = reqs.pop(best_idx)
            scheduled.append(chosen)
            cache_tree.insert(chosen)
        return scheduled
    return list(requests)
