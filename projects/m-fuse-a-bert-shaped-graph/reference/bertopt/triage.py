def triage_zero_attention(graph_def):
    nodes = graph_def.get("nodes", [])
    has_zero = any(n.get("op") == "Attention" and n.get("zero_head", False) for n in nodes)
    if has_zero:
        return {"status": "triaged_safe", "action": "bypass_or_patch"}
    return {"status": "standard", "action": "optimize"}
