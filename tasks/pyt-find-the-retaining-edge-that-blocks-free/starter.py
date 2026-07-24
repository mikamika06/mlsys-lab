def find_retaining_edge(graph):
    # TODO: only checks obvious list references and misses attribute and
    # closure-cell retaining edges.
    for edge in graph["edges"]:
        if edge["kind"] == "list":
            return edge["id"]
    return None
