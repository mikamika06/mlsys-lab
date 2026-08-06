def compute_copy_cost(partitions, tensor_metadata):
    total_cost = 0.0
    for p in partitions:
        for t_id in p.get("boundary_tensors", []):
            shape = tensor_metadata.get(t_id, [1])
            elements = 1
            for dim in shape:
                elements *= dim
            total_cost += float(elements * 4)
    return total_cost
