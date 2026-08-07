def check(workdir):
    import numpy as np
    import speculation.tree as tree
    import ref

    m = {"mask_correct": 0.0}

    parents = [-1, 0, 1, 1, 0]
    mask = tree.tree_attention_mask(parents)
    expected = ref.tree_attention_mask(parents)

    if np.array_equal(mask, expected):
        m["mask_correct"] = 1.0

    return m
