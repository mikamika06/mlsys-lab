from radix.tree import TokenRadixTree

def schedule_requests(requests, policy="fcfs"):
    if policy == "fcfs":
        return list(requests)
    elif policy == "lpm":
        tree = TokenRadixTree()
        remaining = list(requests)
        scheduled = []
        while remaining:
            best_idx = 0
            best_len = -1
            for i, req in enumerate(remaining):
                matched, _ = tree.longest_match(req)
                if len(matched) > best_len:
                    best_len = len(matched)
                    best_idx = i
            chosen = remaining.pop(best_idx)
            scheduled.append(chosen)
            tree.insert(chosen)
        return scheduled
    return list(requests)
