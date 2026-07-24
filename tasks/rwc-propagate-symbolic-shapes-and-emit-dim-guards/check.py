def _oracle(graph):
    tensors = {}
    guards = []

    def prod(xs):
        if not xs:
            return "1"
        out = xs[0]
        for x in xs[1:]:
            out = out + "*" + x
        return out

    def normalize_product(xs):
        return prod(xs)

    for node in graph:
        op = node["op"]
        if op == "input":
            tensors[node["output"]] = list(node["shape"])
        elif op in ("reshape", "view"):
            src = tensors[node["inputs"][0]]
            target = list(node["shape"])
            old_numel = normalize_product(src)
            missing = None
            known = []
            for d in target:
                if d == "-1":
                    missing = len(known)
                else:
                    known.append(d)
            if missing is not None:
                known_numel = normalize_product(known)
                target[missing] = old_numel + "/" + known_numel
                guards.append(old_numel + " % " + known_numel + " == 0")
            else:
                new_numel = normalize_product(target)
                guards.append(old_numel + " == " + new_numel)
            tensors[node["output"]] = target
        elif op == "cat":
            shapes = [tensors[x] for x in node["inputs"]]
            axis = node["axis"]
            out = list(shapes[0])
            parts = [s[axis] for s in shapes]
            out[axis] = "+".join(parts)
            tensors[node["output"]] = out
        elif op == "matmul":
            a = tensors[node["inputs"][0]]
            b = tensors[node["inputs"][1]]
            guards.append(a[1] + " == " + b[0])
            tensors[node["output"]] = [a[0], b[1]]

    return {
        "shapes": {k: ",".join(v) for k, v in tensors.items()},
        "guards": sorted(set(guards)),
    }


def grade(sol, fx) -> dict:
    cases = [
        [
            {"op": "input", "output": "x", "shape": ["s0", "4"]},
            {"op": "reshape", "inputs": ["x"], "output": "y", "shape": ["8", "-1"]},
        ],
        [
            {"op": "input", "output": "a", "shape": ["s0", "s1"]},
            {"op": "input", "output": "b", "shape": ["s1", "32"]},
            {"op": "matmul", "inputs": ["a", "b"], "output": "c"},
        ],
        [
            {"op": "input", "output": "a", "shape": ["s0", "4"]},
            {"op": "input", "output": "b", "shape": ["s1", "4"]},
            {"op": "cat", "inputs": ["a", "b"], "axis": 0, "output": "c"},
            {"op": "view", "inputs": ["c"], "output": "d", "shape": ["8", "-1"]},
        ],
    ]
    ok = 1.0
    for graph in cases:
        try:
            got = sol.propagate_shapes(graph)
            got = {
                "shapes": dict(got["shapes"]),
                "guards": sorted(set(got["guards"])),
            }
        except Exception:
            ok = 0.0
            break
        if got != _oracle(graph):
            ok = 0.0
            break
    return {"exact_match": ok}
