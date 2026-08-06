def analyze_trace_memcpy(events, threshold_ratio=0.3):
    total_dur = sum(e.get("dur", 0) for e in events if e.get("ph") == "X")
    memcpy_dur = 0
    compute_dur = 0
    node_stats = {}

    for e in events:
        if e.get("ph") != "X":
            continue
        dur = e.get("dur", 0)
        name = e.get("name", "")
        cat = e.get("cat", "")
        op_type = e.get("args", {}).get("op_name", name)
        is_copy = (
            "memcpy" in name.lower()
            or "memcpy" in cat.lower()
            or op_type in ("Memcpy", "Identity", "Reshape", "Transpose")
            and e.get("args", {}).get("provider") == "CPUExecutionProvider"
            and "copy" in name.lower()
        )
        if is_copy or op_type in ("MemcpyFromHost", "MemcpyToHost", "Memcpy"):
            memcpy_dur += dur
        else:
            compute_dur += dur

        if op_type not in node_stats:
            node_stats[op_type] = {"count": 0, "total_dur": 0}
        node_stats[op_type]["count"] += 1
        node_stats[op_type]["total_dur"] += dur

    memcpy_ratio = (memcpy_dur / total_dur) if total_dur > 0 else 0.0
    is_dominated = memcpy_ratio >= threshold_ratio

    return {
        "total_duration": total_dur,
        "memcpy_duration": memcpy_dur,
        "compute_duration": compute_dur,
        "memcpy_ratio": memcpy_ratio,
        "is_memcpy_dominated": is_dominated,
        "node_stats": node_stats,
    }


def find_memcpy_root_causes(events):
    root_causes = []
    for e in events:
        if e.get("ph") != "X":
            continue
        op_name = e.get("args", {}).get("op_name", e.get("name", ""))
        args = e.get("args", {})
        if op_name in ("Memcpy", "MemcpyFromHost", "MemcpyToHost") or "memcpy" in e.get("name", "").lower():
            cause = {
                "node_name": e.get("name"),
                "dur": e.get("dur", 0),
                "reason": args.get("reason", "non_contiguous_layout"),
                "src_node": args.get("src_node", "unknown"),
                "dst_node": args.get("dst_node", "unknown"),
            }
            root_causes.append(cause)
    root_causes.sort(key=lambda x: x["dur"], reverse=True)
    return root_causes
