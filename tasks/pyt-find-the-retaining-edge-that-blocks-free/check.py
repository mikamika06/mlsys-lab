import weakref


def _cut(edge):
    kind = edge["kind"]
    if kind == "list":
        owner = edge["owner"]
        slot = edge["slot"]
        old = owner[slot]
        owner[slot] = None
        return lambda: owner.__setitem__(slot, old)
    if kind == "attribute":
        owner = edge["owner"]
        name = edge["name"]
        old = getattr(owner, name)
        setattr(owner, name, None)
        return lambda: setattr(owner, name, old)
    if kind == "cell":
        cell = edge["cell"]
        old = cell.cell_contents
        cell.cell_contents = None
        return lambda: setattr(cell, "cell_contents", old)
    raise ValueError("unknown edge")


def _oracle(graph):
    target = graph["target"]
    for edge in graph["edges"]:
        ref = weakref.ref(target)
        restore = _cut(edge)
        try:
            alive = ref() is not None
        finally:
            restore()
        if not alive:
            return edge["id"]
    return None


def _make_cell(value):
    def outer():
        x = value
        return lambda: x
    fn = outer()
    return fn.__closure__[0]


def grade(sol, fx) -> dict:
    class Box:
        pass

    class Holder:
        pass

    cases = []

    a = Box()
    lst = [a, "noise"]
    cases.append({
        "target": a,
        "edges": [
            {"id": "list-slot-0", "kind": "list", "owner": lst, "slot": 0}
        ],
    })

    b = Box()
    holder = Holder()
    holder.keep = "temporary"
    holder.keep = b
    cases.append({
        "target": b,
        "edges": [
            {"id": "attribute-keep", "kind": "attribute", "owner": holder, "name": "keep"}
        ],
    })

    c = Box()
    cell = _make_cell(c)
    cases.append({
        "target": c,
        "edges": [
            {"id": "closure-cell", "kind": "cell", "cell": cell}
        ],
    })

    d = Box()
    first = [d]
    second = Holder()
    second.value = d
    cases.append({
        "target": d,
        "edges": [
            {"id": "distractor", "kind": "list", "owner": first, "slot": 0},
            {"id": "blocking-edge", "kind": "attribute", "owner": second, "name": "value"},
        ],
    })

    ok = 1.0
    for graph in cases:
        expected = _oracle(graph)
        try:
            got = sol.find_retaining_edge(graph)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
