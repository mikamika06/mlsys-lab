def greedy_fuse(ops):
    groups = []
    current = []
    for op in ops:
        if op.startswith("pointwise"):
            current.append(op)
        else:
            if current:
                groups.append(current)
                current = []
            groups.append([op])
    if current:
        groups.append(current)
    return groups
