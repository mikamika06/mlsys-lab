class _Node:
    __slots__ = ("children",)

    def __init__(self):
        self.children = {}


def _insert(root, seq):
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


def branch_savings(trunk: list, continuations: list) -> int:
    """
    trunk: shared prefix token list.
    continuations: list of K token lists; branch i's full sequence is
        trunk + continuations[i].

    Insert the K full sequences into ONE shared radix/prefix tree, in
    order, exactly like a real prefix-cache / RadixAttention tree would.
    Return the total number of tokens saved across all branches: for each
    branch, the number of tokens at the start of its sequence that were
    already present in the tree from an earlier branch (the trunk is only
    "new" once -- paid for by the first branch that inserts it -- and
    every later branch reusing it, plus any deeper shared structure among
    branches, counts toward the total).
    """
    root = _Node()
    total_saved = 0
    for cont in continuations:
        full = list(trunk) + list(cont)
        total_saved += _insert(root, full)
    return total_saved
