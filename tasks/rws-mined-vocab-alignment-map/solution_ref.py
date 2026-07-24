import numpy as np


def _edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int64)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
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
        out.append(int(np.argmin(dists)))
    return out
