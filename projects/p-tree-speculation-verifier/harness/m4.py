import random


def check(workdir):
    import speculation.tree as tree
    import ref

    m = {"exact_distribution": 0.0}

    tokens = [1, 2, 1, 3]
    parents = [-1, -1, 0, 0]
    draft_probs = [0.5, 0.5, 0.25, 0.75]
    target_probs = [
        [0.125, 0.375, 0.25, 0.0, 0.25],
        [0.0, 0.25, 0.0, 0.75, 0.0],
        [0.25, 0.25, 0.25, 0.25, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0]
    ]

    rng = random.Random(42)
    matches = 0
    total = 50
    for _ in range(total):
        node_r = [rng.random() for _ in range(5)]
        resample_r = [rng.random() for _ in range(5)]

        acc_learner = tree.verify_tree(tokens, parents, draft_probs, target_probs, node_r, resample_r)
        acc_ref = ref.verify_tree(tokens, parents, draft_probs, target_probs, node_r, resample_r)

        if acc_learner == acc_ref:
            matches += 1

    if matches == total:
        m["exact_distribution"] = 1.0

    return m
