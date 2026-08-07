def check(workdir):
    import speculation.tree as tree
    import ref

    m = {"calculates_expected_length": 0.0, "tree_beats_linear": 0.0}

    tokens = [1, 2, 3]
    parents = [-1, 0, 0]
    draft_probs = [0.875, 0.25, 0.5]
    target_probs = [
        [0.125, 0.875, 0.0, 0.0, 0.0],
        [0.125, 0.125, 0.25, 0.5, 0.0],
        [0.125, 0.125, 0.125, 0.125, 0.5],
        [0.125, 0.125, 0.125, 0.125, 0.5],
    ]

    el = tree.expected_length(tokens, parents, draft_probs, target_probs)
    ref_el = ref.expected_length(tokens, parents, draft_probs, target_probs)

    if abs(el - ref_el) < 1e-5:
        m["calculates_expected_length"] = 1.0

    tokens_lin = [1, 3]
    parents_lin = [-1, 0]
    draft_probs_lin = [0.875, 0.5]
    target_probs_lin = [
        [0.125, 0.875, 0.0, 0.0, 0.0],
        [0.125, 0.125, 0.25, 0.5, 0.0],
        [0.125, 0.125, 0.125, 0.125, 0.5],
    ]
    el_lin = tree.expected_length(tokens_lin, parents_lin, draft_probs_lin, target_probs_lin)

    if el > el_lin:
        m["tree_beats_linear"] = 1.0

    return m
