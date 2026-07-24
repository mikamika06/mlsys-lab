def infer_auto_device_map(modules, max_memory, no_split_modules):
    sizes = {name: size for name, size in modules}

    group_of = {}
    for group in no_split_modules:
        for name in group:
            group_of[name] = tuple(group)

    blocks = []
    seen = set()

    for index, (name, size) in enumerate(modules):
        if name in seen:
            continue
        if name in group_of:
            names = list(group_of[name])
            total = sum(sizes[item] for item in names)
            blocks.append((index, names, total))
            seen.update(names)
        else:
            blocks.append((index, [name], size))
            seen.add(name)

    blocks.sort(key=lambda item: item[0])

    remaining = dict(max_memory)
    result = {}

    for _, names, size in blocks:
        device = None
        for candidate in remaining:
            if remaining[candidate] >= size:
                device = candidate
                break
        if device is None:
            raise ValueError("block does not fit")
        remaining[device] -= size
        for name in names:
            result[name] = device

    return result
