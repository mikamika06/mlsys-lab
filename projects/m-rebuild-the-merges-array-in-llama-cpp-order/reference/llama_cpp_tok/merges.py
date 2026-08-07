def rebuild_merges(merges):
    indexed = []
    for item in merges:
        if isinstance(item, str):
            parts = item.strip().split()
            if len(parts) >= 2:
                indexed.append((parts[0], parts[1]))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            indexed.append((str(item[0]), str(item[1])))
    indexed.sort(key=lambda x: (len(x[0]) + len(x[1]), x[0], x[1]))
    return [f"{a} {b}" for a, b in indexed]
