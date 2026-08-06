import ref


def check(workdir):
    from delpipeline import classifier, postmortem

    graphs = ref.generate_test_graphs()
    # pick one graph where delegation fails completely (zero supported ops)
    zero_graph = {
        "graph_id": 99,
        "ops": [{"id": k, "name": op, "shape": [1, 8, 8, 3], "dtype": "FLOAT32", "supported": False} for k, op in enumerate(ref.OPS_15[:5])]
    }

    support = classifier.classify_support(zero_graph)
    analysis = postmortem.analyze_zero_node(zero_graph, support)

    match = 1 if isinstance(analysis, dict) and "reasons" in analysis and len(analysis["reasons"]) > 0 else 0
    return {"postmortem_match": match}
