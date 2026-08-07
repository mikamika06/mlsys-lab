def import_torch_mlp(exported_program):
    """Dummy or pure python representation parser simulating torch export import structure."""
    nodes = getattr(exported_program, "graph_nodes", [])
    call_tir_count = 0
    raw_ops_count = 0
    for node in nodes:
        if node == "call_tir":
            call_tir_count += 1
        elif node == "raw_op":
            raw_ops_count += 1
    return {
        "mod": exported_program,
        "call_tir_count": call_tir_count,
        "raw_ops_count": raw_ops_count
    }
