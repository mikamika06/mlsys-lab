def classify_snippets(snippets):
    """Classify snippets as pass, graph_break, or error under default and fullgraph modes."""
    results = []
    for s in snippets:
        snip_id = s["id"]
        invalid = s.get("has_invalid_shape_or_type", False)
        break_cond = s.get("has_data_dependent_branch", False) or s.get("has_unsupported_side_effect", False)

        if invalid:
            default_res = "error"
            fullgraph_res = "error"
        elif break_cond:
            default_res = "graph_break"
            fullgraph_res = "error"
        else:
            default_res = "pass"
            fullgraph_res = "pass"

        results.append({
            "id": snip_id,
            "default": default_res,
            "fullgraph": fullgraph_res
        })
    return results
