def count_fused_nodes(graph_nodes):
    """Count nodes belonging to the com.microsoft domain."""
    count = 0
    for node in graph_nodes:
        domain = node.get("domain", "")
        op_type = node.get("op_type", "")
        if domain == "com.microsoft" or op_type.startswith("com.microsoft:"):
            count += 1
    return count
