def propagate_shapes(graph):
    tensors = {}
    guards = []

    def prod(xs):
        if not xs:
            return "1"
        value = xs[0]
        for x in xs[1:]:
            value += "*" + x
        return value

    for node in graph:
        op = node["op"]
        if op == "input":
            tensors[node["output"]] = list(node["shape"])
        elif op in ("reshape", "view"):
            src = tensors[node["inputs"][0]]
            target = list(node["shape"])
            old = prod(src)
            unknown = -1
            known = []
            for d in target:
                if d == "-1":
                    unknown = len(known)
                else:
                    known.append(d)
            if unknown != -1:
                known_value = prod(known)
                target[unknown] = old + "/" + known_value
                guards.append(old + " % " + known_value + " == 0")
            else:
                guards.append(old + " == " + prod(target))
            tensors[node["output"]] = target
        elif op == "cat":
            values = [tensors[name] for name in node["inputs"]]
            axis = node["axis"]
            result = list(values[0])
            result[axis] = "+".join(x[axis] for x in values)
            tensors[node["output"]] = result
        elif op == "matmul":
            left = tensors[node["inputs"][0]]
            right = tensors[node["inputs"][1]]
            guards.append(left[1] + " == " + right[0])
            tensors[node["output"]] = [left[0], right[1]]

    return {
        "shapes": {name: ",".join(shape) for name, shape in tensors.items()},
        "guards": sorted(set(guards)),
    }
