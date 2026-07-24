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


def find_retaining_edge(graph):
    import weakref

    target = graph["target"]
    for edge in graph["edges"]:
        ref = weakref.ref(target)
        restore = _cut(edge)
        try:
            if ref() is None:
                return edge["id"]
        finally:
            restore()
    return None
