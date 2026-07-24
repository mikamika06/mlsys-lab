import random


def _oracle(seqs):
    """Reference radix/prefix tree: insert sequences one at a time, and
    for each one walk from the root as far as an exact matching path
    already exists (those tokens are reused/"saved"), then graft the
    remaining tokens on as new tree nodes so later sequences can reuse
    them too."""
    root = {}
    total_saved = 0
    total_computed = 0

    for seq in seqs:
        node = root
        i = 0
        while i < len(seq) and seq[i] in node:
            node = node[seq[i]]
            i += 1

        total_saved += i
        total_computed += len(seq) - i

        for j in range(i, len(seq)):
            child = {}
            node[seq[j]] = child
            node = child

    return total_saved, total_computed


def _cases():
    cases = []

    # Hand-picked: exact duplicate, an extension, a partial-prefix
    # divergence, and a fully novel sequence.
    cases.append(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],        # exact duplicate -> fully saved
            [1, 2, 3, 4, 5, 6, 7],  # extension -> first 5 saved, 2 computed
            [1, 2, 9, 9, 9],        # diverges after 2 tokens
            [100, 200],             # totally novel
        ]
    )

    # Empty sequence edge case mixed in.
    cases.append([[], [1], [1, 2], []])

    # Deterministic pseudo-random cases: a handful of shared "system
    # prompt" prefixes, each followed by varying random suffixes, plus
    # some fully independent sequences.
    rng = random.Random(0)
    for seed in [1, 2, 3]:
        r = random.Random(seed)
        shared_prefixes = [
            [r.randint(0, 50) for _ in range(r.randint(3, 8))] for _ in range(3)
        ]
        seqs = []
        for _ in range(12):
            if r.random() < 0.7:
                prefix = r.choice(shared_prefixes)
            else:
                prefix = []
            suffix_len = r.randint(0, 6)
            suffix = [r.randint(1000, 1010) for _ in range(suffix_len)]
            seqs.append(prefix + suffix)
        cases.append(seqs)

    return cases


def grade(sol, fx) -> dict:
    total = 0
    correct = 0
    for seqs in _cases():
        total += 1
        ref = _oracle(seqs)
        try:
            got = sol.prefix_tokens_saved([list(s) for s in seqs])
            got = tuple(int(x) for x in got)
        except Exception:
            continue
        if got == ref:
            correct += 1

    return {"exact_match": correct / total if total else 0.0}
