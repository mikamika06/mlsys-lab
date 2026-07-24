def _oracle(corpus, num_merges):
    work = [list(seq) for seq in corpus]
    merges = []

    for _ in range(num_merges):
        counts = {}
        for seq in work:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] = counts.get(pair, 0) + 1

        if not counts:
            break

        best = min(counts, key=lambda p: (-counts[p], p))
        merges.append(best)

        left, right = best
        merged = left + right
        updated = []
        for seq in work:
            out = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == left and seq[i + 1] == right:
                    out.append(merged)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            updated.append(out)
        work = updated

    return merges


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                ["l", "o", "w"],
                ["l", "o", "w"],
                ["l", "o", "w", "er"],
            ],
            4,
        ),
        (
            [
                ["a", "b", "a", "b"],
                ["a", "b", "c"],
            ],
            3,
        ),
        (
            [
                ["x", "y"],
                ["x", "z"],
                ["y", "z"],
            ],
            5,
        ),
        (
            [
                ["t", "h", "e"],
                ["t", "h", "e", "r", "e"],
            ],
            6,
        ),
    ]

    ok = 1.0
    for corpus, k in cases:
        expected = _oracle(corpus, k)
        try:
            got = sol.train_bpe_merges(corpus, k)
            got = [tuple(x) for x in got]
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
