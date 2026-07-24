def _oracle_wengert(output_node):
    seen = set()
    order = []

    def visit(node):
        ident = id(node)
        if ident in seen:
            return
        seen.add(ident)
        for inp in getattr(node, "inputs", []):
            visit(inp)
        order.append(
            {
                "name": node.name,
                "op": node.op,
                "inputs": [x.name for x in getattr(node, "inputs", [])],
            }
        )

    visit(output_node)
    return order


class _Node:
    def __init__(self, name, op, inputs=None):
        self.name = name
        self.op = op
        self.inputs = [] if inputs is None else inputs


def grade(sol, fx) -> dict:
    cases = []

    a = _Node("a", "leaf")
    b = _Node("b", "leaf")
    c = _Node("c", "mul", [a, b])
    d = _Node("d", "add", [c, a])
    cases.append(d)

    x = _Node("x", "leaf")
    y = _Node("y", "leaf")
    z = _Node("z", "leaf")
    xy = _Node("xy", "add", [x, y])
    out = _Node("out", "mul", [xy, z])
    cases.append(out)

    p = _Node("p", "leaf")
    q = _Node("q", "leaf")
    r = _Node("r", "add", [p, q])
    s = _Node("s", "mul", [r, q])
    t = _Node("t", "add", [s, p])
    cases.append(t)

    expected = [_oracle_wengert(x) for x in cases]

    try:
        got = [sol.build_wengert_list(x) for x in cases]
    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0 if got == expected else 0.0}
