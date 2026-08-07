import ref

def check(workdir):
    from profops.rank import get_top_ops
    out = {"recall_at_k": 0.0}
    scores = []
    for rows in ref.PROFILES:
        want = ref.top_k_ops(rows, 5)
        got = get_top_ops(rows, 5)
        scores.append(ref.recall_at_k(want, got, 5))
    avg_score = sum(scores) / len(scores) if scores else 0.0
    out["recall_at_k"] = float(avg_score >= 1.0)
    return out
