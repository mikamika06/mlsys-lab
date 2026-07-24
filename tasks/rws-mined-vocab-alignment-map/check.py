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


def _oracle(teacher, student):
    out = []
    for t in teacher:
        dists = [_edit_distance(t, s) for s in student]
        out.append(int(np.argmin(dists)))  # np.argmin -> first minimum on ties
    return out


_ALPHABET = list("abcde")


def _rand_tok(rng, lo=2, hi=6):
    n = int(rng.integers(lo, hi))
    return "".join(rng.choice(_ALPHABET, size=n))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ok = 1.0

    for _ in range(6):
        n_t = int(rng.integers(5, 12))
        n_s = int(rng.integers(5, 12))
        teacher = [_rand_tok(rng) for _ in range(n_t)]
        student = [_rand_tok(rng) for _ in range(n_s)]
        # force at least one exact duplicate to guarantee tie-break coverage
        student[0] = teacher[0]
        if n_s > 1:
            student[1] = teacher[0]

        exp = _oracle(teacher, student)

        try:
            got = list(sol.mined_vocab_align(list(teacher), list(student)))
            got = [int(v) for v in got]
        except Exception:
            ok = 0.0
            continue

        if got != exp:
            ok = 0.0

    return {"exact_match": ok}
