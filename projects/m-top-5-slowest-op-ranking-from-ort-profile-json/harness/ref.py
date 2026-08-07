def generate_profile_data(seed=42):
    import random
    rng = random.Random(seed)
    events = []
    events.append({"cat": "Profiling", "name": "ProfilerStart", "dur": 100})
    events.append({"cat": "Profiling", "name": "ProfilerEnd", "dur": 150})

    ops = [
        ("MatMul", "Gemm_1", 50000),
        ("MatMul", "Gemm_2", 40000),
        ("Add", "Add_1", 5000),
        ("Mul", "Mul_1", 3000),
        ("LayerNorm", "Norm_1", 20000),
        ("Softmax", "Softmax_1", 10000),
        ("Relu", "Relu_1", 2000),
        ("ReduceMean", "Reduce_1", 4000)
    ]

    for op_type, name, base_dur in ops:
        for i in range(rng.randint(2, 5)):
            dur = base_dur + rng.randint(-500, 500)
            events.append({
                "cat": "Node",
                "args": {"op_name": op_type},
                "name": f"{name}_{i}",
                "dur": dur,
                "ph": "X"
            })
    return events


def rank_top_slowest(events):
    totals = {}
    for e in events:
        if e.get("cat") == "Node":
            op = e.get("args", {}).get("op_name", "Unknown")
            totals[op] = totals.get(op, 0) + e.get("dur", 0)
    sorted_ops = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    return [op for op, dur in sorted_ops[:5]]


def compute_overhead(events):
    node_time = sum(e.get("dur", 0) for e in events if e.get("cat") == "Node")
    prof_time = sum(e.get("dur", 0) for e in events if e.get("cat") == "Profiling")
    total_time = node_time + prof_time
    if total_time == 0:
        return 0.0
    return float(prof_time) / float(total_time)


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
