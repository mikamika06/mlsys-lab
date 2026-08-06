def compute_fusion_cost(break_info, tensor_shapes):
    total_bytes = 0
    fused_ops_lost = 0
    for b in break_info:
        shape = tensor_shapes.get(b["index"], [1, 1])
        elements = 1
        for dim in shape:
            elements *= dim
        bytes_per_elem = 4
        total_bytes += elements * bytes_per_elem
        fused_ops_lost += b.get("ops_before", 1) * b.get("ops_after", 1)
    return {"materialized_bytes": total_bytes, "lost_fusion_score": fused_ops_lost}
