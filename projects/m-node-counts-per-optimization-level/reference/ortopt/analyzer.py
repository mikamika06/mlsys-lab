def count_nodes(model_desc, level):
    nodes = list(model_desc.get("nodes", []))
    base = len(nodes)
    if level == 0:
        return base
    if level == 1:
        fused = [n for n in nodes if not (n.get("type") in ("Add", "Mul") and n.get("fusable"))]
        return max(1, len(fused))
    if level >= 99:
        return max(1, base // 3)
    return base
