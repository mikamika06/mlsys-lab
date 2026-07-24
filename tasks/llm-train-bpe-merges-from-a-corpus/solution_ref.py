def train_bpe_merges(corpus, num_merges):
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
        combined = left + right
        new_work = []

        for seq in work:
            out = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == left and seq[i + 1] == right:
                    out.append(combined)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            new_work.append(out)

        work = new_work

    return merges
