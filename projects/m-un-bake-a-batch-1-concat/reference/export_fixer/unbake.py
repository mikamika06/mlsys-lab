def unbake_concat(graph_dict):
    nodes = graph_dict.get("node", [])
    fixed_nodes = []
    for node in nodes:
        if node.get("op") == "Concat":
            attrs = node.get("attribute", [])
            new_attrs = []
            for attr in attrs:
                if attr.get("name") == "axis" and attr.get("i") == 0:
                    attr = dict(attr)
                    attr["_unbaked_batch_one"] = True
                new_attrs.append(attr)
            node = dict(node)
            node["attribute"] = new_attrs
        fixed_nodes.append(node)
    return {"node": fixed_nodes, "unbaked": True}
