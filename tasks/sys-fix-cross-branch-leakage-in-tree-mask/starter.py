import numpy as np


def build_tree_mask(parents):
    # TODO: this leaks sibling branches by allowing every earlier token.
    # A speculative token should only see its ancestors, not all previous nodes.
    n = len(parents)
    mask = np.zeros((n, n), dtype=np.int8)

    for i in range(n):
        mask[i, : i + 1] = 1

    return mask
