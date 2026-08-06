def count_launches(op_sequence_len, steps, use_graph):
    """Count host dispatches vs graph launches for a sequence of ops."""
    if steps <= 0 or op_sequence_len < 0:
        return {"total_dispatches": 0, "graph_launches": 0, "individual_launches": 0}

    if use_graph:
        return {
            "total_dispatches": steps,
            "graph_launches": steps,
            "individual_launches": 0,
        }

    total = op_sequence_len * steps
    return {
        "total_dispatches": total,
        "graph_launches": 0,
        "individual_launches": total,
    }
