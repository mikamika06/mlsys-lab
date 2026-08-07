def _mul(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return [_mul(x, y) for x, y in zip(a, b)]
    return a * b


def _add(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return [_add(x, y) for x, y in zip(a, b)]
    return a + b


def fold_constants(nodes):
    values = {}
    known = {}
    folded = 0

    for node in nodes:
        name = node["name"]
        op = node["op"]

        if op == "const":
            values[name] = node["value"]
            known[name] = True
        elif op == "mul":
            a, b = node["inputs"]
            values[name] = _mul(values[a], values[b])
            known[name] = known[a] and known[b]
            if known[name]:
                folded += 1
        elif op == "add":
            a, b = node["inputs"]
            values[name] = _add(values[a], values[b])
            known[name] = known[a] and known[b]
            if known[name]:
                folded += 1
        elif op == "identity":
            src = node["inputs"][0]
            values[name] = values[src]
            known[name] = known[src]
            if known[name]:
                folded += 1

    return values[nodes[-1]["name"]], folded
