def parse_fusion_trace(log_text):
    fused_groups = []
    separate_ops = []

    lines = [line.strip() for line in log_text.strip().split("\n") if line.strip()]
    for line in lines:
        if line.startswith("FUSED:"):
            body = line[len("FUSED:"):].strip()
            ops = [op.strip() for op in body.split(",") if op.strip()]
            if ops:
                fused_groups.append(ops)
        elif line.startswith("SEPARATE:"):
            body = line[len("SEPARATE:"):].strip()
            ops = [op.strip() for op in body.split(",") if op.strip()]
            separate_ops.extend(ops)

    return {
        "fused_groups": fused_groups,
        "separate_ops": separate_ops,
    }
