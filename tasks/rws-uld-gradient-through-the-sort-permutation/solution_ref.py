def uld_gradient(student_logits: list[float], teacher_logits: list[float]) -> list[float]:
    """Gradient of ULD loss w.r.t. student_logits."""
    n = len(student_logits)

    s_indexed = sorted([(student_logits[i], i) for i in range(n)], key=lambda x: x[0])
    t_sorted = sorted([teacher_logits[i] for i in range(n)])

    rank = [0] * n
    for k in range(n):
        orig_idx = s_indexed[k][1]
        rank[orig_idx] = k

    result = [2.0 * (student_logits[i] - t_sorted[rank[i]]) for i in range(n)]
    return result
