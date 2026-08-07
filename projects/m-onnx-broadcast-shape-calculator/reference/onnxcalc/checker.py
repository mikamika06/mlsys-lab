from onnxcalc.value_info import infer_graph_value_info


def triage_graph(graph):
    try:
        val_info = infer_graph_value_info(graph)
        return {"valid": True, "errors": [], "value_info": val_info}
    except Exception as e:
        return {"valid": False, "errors": [str(e)], "value_info": {}}
