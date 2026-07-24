def identity_partition(objects):
    groups = {}
    for index, obj in enumerate(objects):
        key = id(obj)
        if key not in groups:
            groups[key] = []
        groups[key].append(index)

    return tuple(
        tuple(indices)
        for indices in sorted(groups.values(), key=lambda group: group[0])
    )
