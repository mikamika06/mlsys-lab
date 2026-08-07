def count_relax_ops(mod):
    """Counts call_tir ops and raw relax ops in an IRModule construct."""
    if isinstance(mod, dict):
        return {
            "call_tir": mod.get("call_tir_count", 0),
            "raw_ops": mod.get("raw_ops_count", 0)
        }
    nodes = getattr(mod, "graph_nodes", [])
    call_tir = sum(1 for n in nodes if n == "call_tir")
    raw_ops = sum(1 for n in nodes if n == "raw_op")
    return {"call_tir": call_tir, "raw_ops": raw_ops}
