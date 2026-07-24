import numpy as np


def _oracle(seqs) -> bool:
    owners = {}
    for seq in seqs:
        ids = seq["physical_block_ids"]
        shared = seq["is_shared"]
        seen_this_seq = set()
        for pid, is_shared in zip(ids, shared):
            if pid in seen_this_seq:
                continue
            seen_this_seq.add(pid)
            owners.setdefault(pid, []).append(bool(is_shared))

    for pid, flags in owners.items():
        if len(flags) > 1 and not all(flags):
            return False
    return True


def _handcrafted_cases():
    cases = []

    # 1: two disjoint sequences, no overlap at all -> legal
    cases.append([
        {"physical_block_ids": [0, 1, 2], "is_shared": [False, False, False]},
        {"physical_block_ids": [3, 4], "is_shared": [False, False]},
    ])

    # 2: proper shared prefix (both mark shared=True) -> legal
    cases.append([
        {"physical_block_ids": [0, 1, 5], "is_shared": [True, True, False]},
        {"physical_block_ids": [0, 1, 6], "is_shared": [True, True, False]},
    ])

    # 3: same as 2 but one side forgot to mark the shared block -> illegal
    cases.append([
        {"physical_block_ids": [0, 1, 5], "is_shared": [True, True, False]},
        {"physical_block_ids": [0, 1, 6], "is_shared": [False, True, False]},
    ])

    # 4: three-way shared prefix, all consistent -> legal
    cases.append([
        {"physical_block_ids": [2, 3], "is_shared": [True, True]},
        {"physical_block_ids": [2, 3, 9], "is_shared": [True, True, False]},
        {"physical_block_ids": [2, 3, 10], "is_shared": [True, True, False]},
    ])

    # 5: three-way shared prefix, ONE of the three has is_shared=False -> illegal
    cases.append([
        {"physical_block_ids": [2, 3], "is_shared": [True, True]},
        {"physical_block_ids": [2, 3, 9], "is_shared": [True, False, False]},
        {"physical_block_ids": [2, 3, 10], "is_shared": [True, True, False]},
    ])

    # 6: two private (writable) sequences accidentally collide on one block -> illegal
    cases.append([
        {"physical_block_ids": [7, 8], "is_shared": [False, False]},
        {"physical_block_ids": [8, 9], "is_shared": [False, False]},
    ])

    # 7: single sequence referencing its own block twice -> legal (not aliasing)
    cases.append([
        {"physical_block_ids": [4, 4, 5], "is_shared": [False, False, False]},
    ])

    # 8: single sequence alone -> always legal
    cases.append([
        {"physical_block_ids": [0, 1, 2, 3], "is_shared": [False, True, False, True]},
    ])

    return cases


def _random_cases():
    rng = np.random.default_rng(53)
    cases = []
    for _ in range(8):
        n_seqs = int(rng.integers(2, 5))
        pool = int(rng.integers(4, 10))
        seqs = []
        for _ in range(n_seqs):
            length = int(rng.integers(1, 5))
            ids = rng.integers(0, pool, size=length).tolist()
            shared = rng.integers(0, 2, size=length).astype(bool).tolist()
            seqs.append({"physical_block_ids": ids, "is_shared": shared})
        cases.append(seqs)
    return cases


def grade(sol, fx) -> dict:
    cases = _handcrafted_cases() + _random_cases()
    total = 0
    correct = 0
    for seqs in cases:
        ref = _oracle(seqs)
        total += 1
        try:
            got = sol.is_block_mapping_legal([dict(s) for s in seqs])
            if bool(got) == ref:
                correct += 1
        except Exception:
            pass
    return {"exact_match": (correct / total) if total else 0.0}
