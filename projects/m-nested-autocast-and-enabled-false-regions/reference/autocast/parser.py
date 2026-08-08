def parse_regions(config):
    def walk(node, depth=0, parent_enabled=True, parent_dtype="float16"):
        enabled = node.get("enabled", parent_enabled)
        dtype = node.get("dtype", parent_dtype)
        res = {
            "id": node["id"],
            "depth": depth,
            "enabled": enabled,
            "dtype": dtype,
            "children": []
        }
        for child in node.get("children", []):
            res["children"].append(walk(child, depth + 1, enabled, dtype))
        return res
    return walk(config)
