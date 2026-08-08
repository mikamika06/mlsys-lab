def evaluate_states(regions):
    results = []
    def traverse(node, active_enabled, active_dtype):
        curr_enabled = node["enabled"]
        curr_dtype = node["dtype"]
        results.append({
            "id": node["id"],
            "effective_enabled": curr_enabled,
            "effective_dtype": curr_dtype
        })
        for child in node["children"]:
            traverse(child, curr_enabled, curr_dtype)
    traverse(regions, True, "float16")
    return results
