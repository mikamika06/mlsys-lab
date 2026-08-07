from symshape.infer import propagate_shapes


def compute_coverage(graph):
    inputs = graph.get("inputs", {})
    nodes = graph.get("nodes", [])
    total_tensors = len(inputs) + len(nodes)

    if total_tensors == 0:
        return {"before": 0.0, "after": 0.0}

    initial_known = sum(1 for s in inputs.values() if s is not None)
    before_ratio = float(initial_known) / float(total_tensors)

    final_shapes = propagate_shapes(graph)
    final_known = sum(1 for s in final_shapes.values() if s is not None)
    after_ratio = float(final_known) / float(total_tensors)

    return {"before": before_ratio, "after": after_ratio}
