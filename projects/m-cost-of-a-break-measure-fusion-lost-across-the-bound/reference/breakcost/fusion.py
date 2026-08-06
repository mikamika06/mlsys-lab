def find_fusion_pairs(nodes):
    pairs = []
    for i in range(len(nodes) - 1):
        if nodes[i].get("is_elementwise", False) and nodes[i + 1].get("is_elementwise", False):
            pairs.append((i, i + 1))
    return pairs
