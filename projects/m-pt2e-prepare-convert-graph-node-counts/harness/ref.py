def make_mock_graph(nodes):
    return {"nodes": [{"op": op, "target": target} for op, target in nodes]}


GRAPHS = [
    {
        "orig": make_mock_graph([("placeholder", "x"), ("call_module", "linear1"), ("output", "out")]),
        "prep": make_mock_graph([("placeholder", "x"), ("call_module", "linear1"), ("call_module", "observer1"), ("output", "out")]),
        "conv": make_mock_graph([("placeholder", "x"), ("call_function", "quantized_linear"), ("output", "out")]),
    },
    {
        "orig": make_mock_graph([("placeholder", "input"), ("call_module", "conv1"), ("call_module", "relu1"), ("output", "out")]),
        "prep": make_mock_graph([("placeholder", "input"), ("call_module", "conv1"), ("call_module", "obs_conv"), ("call_module", "relu1"), ("output", "out")]),
        "conv": make_mock_graph([("placeholder", "input"), ("call_function", "q_conv"), ("call_module", "relu1"), ("output", "out")]),
    },
    {
        "orig": make_mock_graph([("placeholder", "a"), ("call_method", "add"), ("output", "out")]),
        "prep": make_mock_graph([("placeholder", "a"), ("call_method", "add"), ("call_module", "obs_add"), ("output", "out")]),
        "conv": make_mock_graph([("placeholder", "a"), ("call_function", "q_add"), ("output", "out")]),
    }
]

from pt2e_counts.analyzer import analyze_graph_counts, compute_conversion_deltas, check_node_invariant
