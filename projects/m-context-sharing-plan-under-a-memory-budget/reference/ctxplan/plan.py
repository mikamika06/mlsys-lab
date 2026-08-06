def build_sharing_plan(tensors, budget):
    sorted_t = sorted(tensors, key=lambda x: x["size"], reverse=True)
    allocated = []
    current_bytes = 0
    for t in sorted_t:
        if current_bytes + t["size"] <= budget:
            allocated.append(t["id"])
            current_bytes += t["size"]
    return sorted(allocated)
