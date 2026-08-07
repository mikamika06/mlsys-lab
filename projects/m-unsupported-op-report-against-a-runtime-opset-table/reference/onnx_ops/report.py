def check_unsupported_ops(graph, opset_table):
    unsupported = []
    for node in graph.get("nodes", []):
        op_type = node.get("op_type")
        domain = node.get("domain", "")
        version = node.get("version", 1)
        supported_versions = opset_table.get((domain, op_type))
        if supported_versions is None or version not in supported_versions:
            unsupported.append({
                "op_type": op_type,
                "domain": domain,
                "version": version
            })
    return sorted(unsupported, key=lambda x: (x["domain"], x["op_type"], x["version"]))
