def build_depgraph(config):
    deps = {}
    layers = {l["name"]: l for l in config["layers"]}

    for l in config["layers"]:
        deps[(l["name"], "out")] = set()
        deps[(l["name"], "in")] = set()

    for l in config["layers"]:
        name = l["name"]
        for inp_name in l.get("inputs", []):
            inp_layer = layers[inp_name]
            if inp_layer["type"] in ("conv", "linear"):
                deps[(inp_name, "out")].add((name, "in"))
                deps[(name, "in")].add((inp_name, "out"))
            elif inp_layer["type"] == "bn":
                deps[(inp_name, "out")].add((name, "in"))
                deps[(name, "in")].add((inp_name, "out"))
            elif inp_layer["type"] == "add":
                deps[(inp_name, "out")].add((name, "in"))
                deps[(name, "in")].add((inp_name, "out"))

    for l in config["layers"]:
        if l["type"] == "bn":
            name = l["name"]
            for inp_name in l.get("inputs", []):
                deps[(inp_name, "out")].add((name, "out"))
                deps[(name, "out")].add((inp_name, "out"))
                deps[(inp_name, "out")].add((name, "in"))
                deps[(name, "in")].add((inp_name, "out"))

    for l in config["layers"]:
        if l["type"] == "add":
            inps = l.get("inputs", [])
            for i in range(len(inps)):
                for j in range(i + 1, len(inps)):
                    n1, n2 = inps[i], inps[j]
                    deps[(n1, "out")].add((n2, "out"))
                    deps[(n2, "out")].add((n1, "out"))
            for inp_name in inps:
                deps[(inp_name, "out")].add((l["name"], "out"))
                deps[(l["name"], "out")].add((inp_name, "out"))

    return deps


def get_pruning_group(config, trigger_layer, prune_channels):
    deps = build_depgraph(config)
    visited = set()
    queue = [(trigger_layer, "out")]
    visited.add((trigger_layer, "out"))

    while queue:
        curr = queue.pop(0)
        for neighbor in deps.get(curr, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    result = {}
    for name, direction in visited:
        if name not in result:
            result[name] = {"in": [], "out": []}
        result[name][direction] = sorted(list(prune_channels))

    return result
