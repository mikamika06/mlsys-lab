import numpy as np


def fold_constants(nodes):
    values = {}
    known = {}
    folded = 0

    for node in nodes:
        name = node["name"]
        op = node["op"]

        if op == "const":
            values[name] = np.asarray(node["value"], dtype=np.float64)
            known[name] = True
        elif op == "mul":
            a, b = node["inputs"]
            values[name] = values[a] * values[b]
            known[name] = known[a] and known[b]
            if known[name]:
                folded += 1
        elif op == "add":
            a, b = node["inputs"]
            values[name] = values[a] + values[b]
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
