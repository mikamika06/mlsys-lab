def migrate_squeeze(node):
    new_node = dict(node)
    attributes = dict(node.get("attributes", {}))
    inputs = list(node.get("inputs", []))
    if "axes" in attributes:
        axes = attributes.pop("axes")
        if len(inputs) < 2:
            inputs.append({"name": node.get("name", "") + "_axes", "values": axes})
        new_node["attributes"] = attributes
        new_node["inputs"] = inputs
    return new_node
