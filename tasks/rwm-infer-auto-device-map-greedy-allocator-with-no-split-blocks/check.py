def _oracle(modules, max_memory, no_split_modules):
    name_to_size = {name: size for name, size in modules}
    group_of = {}
    for group in no_split_modules:
        for name in group:
            group_of[name] = tuple(group)

    blocks = []
    seen = set()
    positions = {name: i for i, (name, _) in enumerate(modules)}

    for index, (name, size) in enumerate(modules):
        if name in seen:
            continue
        if name in group_of:
            group = group_of[name]
            block_size = sum(name_to_size[item] for item in group)
            blocks.append((index, list(group), block_size))
            seen.update(group)
        else:
            blocks.append((index, [name], size))
            seen.add(name)

    blocks.sort(key=lambda item: item[0])

    remaining = dict(max_memory)
    result = {}

    for _, names, size in blocks:
        chosen = None
        for device in remaining:
            if remaining[device] >= size:
                chosen = device
                break
        if chosen is None:
            raise ValueError("block does not fit")
        remaining[chosen] -= size
        for name in names:
            result[name] = chosen

    return result


def grade(sol, fx) -> dict:
    cases = [
        (
            [("a", 40), ("b", 30), ("c", 20)],
            {"d0": 50, "d1": 100},
            [["a", "b"]],
        ),
        (
            [("x0", 10), ("x1", 15), ("x2", 25), ("x3", 5)],
            {"cpu": 30, "gpu": 50},
            [["x1", "x2"]],
        ),
        (
            [("l0", 12), ("l1", 8), ("l2", 9), ("l3", 7), ("l4", 6)],
            {"0": 20, "1": 30, "2": 20},
            [["l0", "l1"], ["l3", "l4"]],
        ),
        (
            [("m0", 5), ("m1", 5), ("m2", 5), ("m3", 10)],
            {"a": 10, "b": 20},
            [["m0", "m1", "m2"]],
        ),
    ]

    for modules, memory, groups in cases:
        try:
            expected = _oracle(modules, memory, groups)
            got = sol.infer_auto_device_map(modules, memory, groups)
        except Exception:
            return {"exact_match": 0.0}
        if got != expected:
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}
