import numpy as np


def _edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int64)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            v1 = dp[i - 1, j] + 1
            v2 = dp[i, j - 1] + 1
            v3 = dp[i - 1, j - 1] + cost
            m_val = v1
            if v2 < m_val:
                m_val = v2
            if v3 < m_val:
                m_val = v3
            dp[i, j] = m_val
    return int(dp[n, m])


def mined_vocab_align(teacher_vocab, student_vocab):
    """
    For each teacher token, return the index of the student token with the
    minimum Levenshtein edit distance to it (ties broken by the smallest
    student index, i.e. first occurrence).
    """
    out = []
    for t in teacher_vocab:
        dists = [_edit_distance(t, s) for s in student_vocab]
        best_idx = 0
        min_val = dists[0]
        for idx in range(1, len(dists)):
            if dists[idx] < min_val:
                min_val = dists[idx]
                best_idx = idx
        out.append(int(best_idx))
    return out
