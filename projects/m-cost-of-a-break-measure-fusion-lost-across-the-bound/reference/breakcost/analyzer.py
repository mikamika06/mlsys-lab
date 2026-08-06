from breakcost.fusion import find_fusion_pairs


def find_lost_fusions(nodes, break_indices):
    all_pairs = find_fusion_pairs(nodes)
    breaks_set = set(break_indices)
    lost = []
    for p in all_pairs:
        if p[0] in breaks_set or p[1] in breaks_set:
            lost.append(p)
    return lost
