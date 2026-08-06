import ref


def check(workdir):
    from breakcost.fusion import find_fusion_pairs
    cases = ref.generate_cases()
    matched = 0
    for nodes, _ in cases:
        want = []
        for i in range(len(nodes) - 1):
            if nodes[i].get("is_elementwise") and nodes[i + 1].get("is_elementwise"):
                want.append((i, i + 1))
        try:
            got = find_fusion_pairs(nodes)
            if list(got) == want:
                matched += 1
        except Exception:
            pass
    return {"fusion_match": float(matched), "_total": float(len(cases))}
