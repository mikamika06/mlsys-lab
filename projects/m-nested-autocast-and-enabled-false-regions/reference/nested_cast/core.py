def parse_tree(node, parent_enabled=False, parent_dtype="float32"):
    enabled = node.get("enabled", parent_enabled)
    dtype = node.get("dtype", parent_dtype)
    res = {
        "enabled": enabled,
        "dtype": dtype,
    }
    if "device_type" in node:
        res["device_type"] = node["device_type"]
    children = [parse_tree(c, enabled, dtype) for c in node.get("children", [])]
    if children:
        res["children"] = children
    return res
