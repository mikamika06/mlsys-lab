def parse_fusion_trace(text: str):
    fused = []
    separate = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("FUSED:"):
            ops = [o.strip() for o in line[len("FUSED:"):].split(",") if o.strip()]
            fused.append(ops)
        elif line.startswith("SEPARATE:"):
            op = line[len("SEPARATE:"):].strip()
            if op:
                separate.append(op)
    return {"fused": fused, "separate": separate}
