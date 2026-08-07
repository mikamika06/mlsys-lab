def reconstruct_eviction_order(snapshots):
    order = []
    seen = set()
    flat_snapshots = [set(s) for s in snapshots]
    for i in range(len(flat_snapshots) - 1):
        current = flat_snapshots[i]
        nxt = flat_snapshots[i + 1]
        diff = current - nxt
        for m in diff:
            if m not in seen:
                seen.add(m)
                order.append(m)
    last_snap = flat_snapshots[-1]
    for m in flat_snapshots[0]:
        if m not in seen and m not in last_snap:
            seen.add(m)
            order.append(m)
    return order
