def compute_separation_savings(tensors, method_a, method_b):
    total_bytes = sum(t["size"] for t in tensors)
    const_bytes = sum(t["size"] for t in tensors if t.get("is_constant", False))

    if method_a == "inline":
        cost_a = total_bytes
    elif method_a == "segmented":
        cost_a = total_bytes - const_bytes + (const_bytes // 2)
    else:
        cost_a = total_bytes

    if method_b == "inline":
        cost_b = total_bytes
    elif method_b == "segmented":
        cost_b = total_bytes - const_bytes + (const_bytes // 2)
    elif method_b == "isolated":
        cost_b = total_bytes - const_bytes
    else:
        cost_b = total_bytes

    return {
        "cost_a": cost_a,
        "cost_b": cost_b,
        "savings_a": total_bytes - cost_a,
        "savings_b": total_bytes - cost_b
    }
