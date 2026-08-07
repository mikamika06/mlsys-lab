def extract_opset_imports(model_dict):
    opsets = {}
    for imp in model_dict.get("opset_import", []):
        domain = imp.get("domain", "")
        version = imp.get("version", 0)
        opsets[domain] = version
    return opsets


def extract_op_inventory(model_dict):
    ops = set()

    def _scan_graph(graph):
        for node in graph.get("nodes", []):
            domain = node.get("domain", "")
            op_type = node.get("op_type", "")
            ops.add((domain, op_type))
            for attr in node.get("attributes", []):
                if "g" in attr and isinstance(attr["g"], dict):
                    _scan_graph(attr["g"])

    graph = model_dict.get("graph", {})
    _scan_graph(graph)
    return sorted(list(ops))
