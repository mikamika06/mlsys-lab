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
    raise NotImplementedError('your code here')
