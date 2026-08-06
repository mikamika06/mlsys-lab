import ref
from breakcost.cost import compute_lost_cost
from breakcost.analyzer import find_lost_fusions


def check(workdir):
    cases = ref.generate_cases()
    ok = 0
    for nodes, breaks in cases:
        all_pairs = []
        for i in range(len(nodes) - 1):
            if nodes[i].get("is_elementwise") and nodes[i + 1].get("is_elementwise"):
                all_pairs.append((i, i + 1))
        b_set = set(breaks)
        want_lost = [p for p in all_pairs if p[0] in b_set or p[1] in b_set]
        want_cost = sum(nodes[p[1]].get("bytes", 0) for p in want_lost)
        try:
            got_lost = find_lost_fusions(nodes, breaks)
            got_cost = compute_lost_cost(nodes, got_lost)
            if got_cost == want_cost:
                ok += 1
        except Exception:
            pass
    return {"cost_match": 1.0 if ok == len(cases) else 0.0}
