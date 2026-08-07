def classify_categories(events):
    categories = {
        "GEMM": ["MatMul", "Gemm", "FusedMatMul"],
        "Normalization": ["LayerNorm", "RMSNorm", "BatchNormalization"],
        "Attention": ["Attention", "MultiHeadAttention"],
        "Elementwise": ["Add", "Mul", "Relu", "Sub", "Div", "Silu"],
        "Reduction": ["ReduceMean", "ReduceSum", "Softmax"]
    }
    shares = {k: 0.0 for k in categories}
    shares["Other"] = 0.0
    total_dur = 0

    cat_map = {}
    for cat_name, op_list in categories.items():
        for op in op_list:
            cat_map[op] = cat_name

    op_totals = {}
    for e in events:
        if e.get("cat") == "Node":
            op = e.get("args", {}).get("op_name", "Unknown")
            dur = e.get("dur", 0)
            op_totals[op] = op_totals.get(op, 0) + dur
            total_dur += dur

    if total_dur == 0:
        return shares

    cat_totals = {}
    for op, dur in op_totals.items():
        c = cat_map.get(op, "Other")
        cat_totals[c] = cat_totals.get(c, 0) + dur

    for c, dur in cat_totals.items():
        shares[c] = float(dur) / float(total_dur)
    return shares
