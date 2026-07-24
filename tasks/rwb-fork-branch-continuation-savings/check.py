import numpy as np


class _Node:
    __slots__ = ("children",)

    def __init__(self):
        self.children = {}


def _insert(root, seq):
    """Insert one token sequence into a shared prefix tree, one token-node
    at a time. Returns how many of its tokens already existed in the tree
    (from an earlier insert) before the first new node had to be created
    -- i.e. the tokens this insert got to reuse instead of paying for."""
    node = root
    matched = 0
    diverged = False
    for tok in seq:
        if not diverged and tok in node.children:
            node = node.children[tok]
            matched += 1
        else:
            diverged = True
            child = _Node()
            node.children[tok] = child
            node = child
    return matched


def _oracle(trunk, continuations):
    root = _Node()
    total_saved = 0
    for cont in continuations:
        full = list(trunk) + list(cont)
        total_saved += _insert(root, full)
    return total_saved


def _hand_cases():
    cases = []

    trunk = [1, 2, 3, 4, 5]
    continuations = [[10, 11], [20, 21, 22], [30], [40, 41, 42, 43]]
    cases.append((trunk, continuations))

    cases.append(([], [[1, 2], [3, 4], [1, 2], [5]]))

    trunk2 = [7, 7, 7]
    continuations2 = [
        [99, 100, 1],
        [99, 100, 2],
        [50],
        [99, 100, 1],
    ]
    cases.append((trunk2, continuations2))

    cases.append(([1, 2, 3], []))

    cases.append(([9, 9], [[1], [1], [1], [1]]))

    return cases


def _gen_case(rng, vocab=200):
    trunk_len = int(rng.integers(0, 12))
    trunk = list(rng.integers(0, vocab, size=trunk_len))
    n_branches = int(rng.integers(2, 6))

    shared_tail_pool = []
    if rng.random() < 0.5:
        shared_tail_pool = list(rng.integers(0, vocab, size=int(rng.integers(1, 4))))

    continuations = []
    for i in range(n_branches):
        parts = []
        if shared_tail_pool and rng.random() < 0.4:
            parts.extend(shared_tail_pool)
        tail_len = int(rng.integers(0, 8))
        parts.extend(list(rng.integers(0, vocab, size=tail_len)))
        continuations.append(parts)
    return trunk, continuations


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = _hand_cases()
    for _ in range(10):
        cases.append(_gen_case(rng))

    exact = 1.0
    for trunk, continuations in cases:
        ref = _oracle(trunk, continuations)
        try:
            got = sol.branch_savings(list(trunk), [list(c) for c in continuations])
        except Exception:
            exact = 0.0
            break
        try:
            got_int = int(got)
        except Exception:
            exact = 0.0
            break
        if got_int != ref:
            exact = 0.0
            break

    return {"exact_match": exact}
