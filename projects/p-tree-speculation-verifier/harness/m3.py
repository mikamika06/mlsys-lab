def check(workdir):
    import speculation.tree as tree
    import ref

    m = {"accepts_correct_branch": 0.0, "stops_on_reject": 0.0}

    tokens = [1, 2, 3]
    parents = [-1, 0, 0]
    draft_probs = [0.875, 0.25, 0.5]
    target_probs = [
        [0.125, 0.875, 0.0, 0.0, 0.0],
        [0.125, 0.125, 0.25, 0.5, 0.0],
        [0.125, 0.125, 0.125, 0.125, 0.5],
        [0.125, 0.125, 0.125, 0.125, 0.5],
    ]

    node_r = [0.0, 0.0, 0.0, 0.0]
    resample_r = [0.0, 0.0, 0.0, 0.0]

    acc = tree.verify_tree(tokens, parents, draft_probs, target_probs, node_r, resample_r)
    ref_acc = ref.verify_tree(tokens, parents, draft_probs, target_probs, node_r, resample_r)

    if acc == ref_acc:
        m["accepts_correct_branch"] = 1.0

    node_r_reject = [0.0, 0.875, 0.0, 0.0]
    acc_reject = tree.verify_tree(tokens, parents, draft_probs, target_probs, node_r_reject, resample_r)
    ref_acc_reject = ref.verify_tree(tokens, parents, draft_probs, target_probs, node_r_reject, resample_r)

    if acc_reject == ref_acc_reject:
        m["stops_on_reject"] = 1.0

    return m
