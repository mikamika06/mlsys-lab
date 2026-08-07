def check_symbolic_propagation(graph, constraints):
    propagated = {}
    for node in graph.get("nodes", []):
        dims = node.get("output_dims", [])
        for d in dims:
            if isinstance(d, str):
                for c in constraints:
                    if c.get("dim_name") == d:
                        propagated[d] = c.get("value")
    return len(propagated) == len(constraints)
