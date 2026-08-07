from trt_precision.formats import analyze_float_formats
from trt_precision.predict import predict_layer_precisions


def run_precision_sweep(graph, execution_mode):
    """Execute precision sweep across FP32, TF32, and FP16 modes."""
    mod_graph = {
        "inputs": [dict(i) for i in graph.get("inputs", [])],
        "nodes": [dict(n) for n in graph.get("nodes", [])]
    }

    if execution_mode == "FORCE_FP16":
        for node in mod_graph["nodes"]:
            if node["op"] != "Cast":
                node["explicit_precision"] = "FP16"
    elif execution_mode == "ALLOW_TF32":
        for node in mod_graph["nodes"]:
            if not node.get("explicit_precision") and node["op"] in ("Conv", "MatMul"):
                node["explicit_precision"] = "TF32"

    layer_precisions = predict_layer_precisions(mod_graph)
    sample_values = graph.get("sample_weights", [1.0, 0.5, 0.125])
    fmt_analysis = analyze_float_formats(sample_values)

    ulp_bounds = {
        node_id: fmt_analysis[prec]["max_ulp"]
        for node_id, prec in layer_precisions.items()
    }

    return {
        "precisions": layer_precisions,
        "ulp_bounds": ulp_bounds,
        "mode": execution_mode
    }
