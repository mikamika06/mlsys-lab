CONFIGS = [
    {
        "device_type": "cuda",
        "children": [
            {"enabled": True, "dtype": "float16", "children": [
                {"enabled": False, "dtype": "float32", "children": []},
                {"enabled": True, "dtype": "bfloat16", "children": []}
            ]}
        ]
    },
    {
        "device_type": "cpu",
        "children": [
            {"enabled": False, "dtype": "bfloat16", "children": [
                {"enabled": True, "dtype": "bfloat16", "children": []}
            ]}
        ]
    },
    {
        "device_type": "cuda",
        "children": [
            {"enabled": True, "dtype": "float16", "children": [
                {"enabled": False, "children": [
                    {"enabled": True, "dtype": "float16", "children": []}
                ]}
            ]}
        ]
    }
]


def resolve_states(config):
    def traverse(node, parent_enabled, parent_dtype):
        enabled = node.get("enabled", parent_enabled)
        dtype = node.get("dtype", parent_dtype)
        res = {"enabled": enabled, "dtype": dtype}
        if "device_type" in node:
            res["device_type"] = node["device_type"]
        children_res = [traverse(c, enabled, dtype) for c in node.get("children", [])]
        if children_res:
            res["children"] = children_res
        return res
    return traverse(config, False, "float32")
