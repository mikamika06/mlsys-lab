import ref


def check(workdir):
    from aot_tools.graph_labeler import label_joint_graph

    out = {"graphs_labeled_correctly": 0.0, "recall_at_k": 0.0}
    total_graphs = len(ref.GRAPHS)
    correct_graphs = 0
    total_nodes = 0
    correct_nodes = 0

    for g in ref.GRAPHS:
        want = ref.label_joint_graph(g)
        got = label_joint_graph(g)

        if not isinstance(got, list) or len(got) != len(want):
            continue

        g_correct = True
        for w, node in zip(want, got):
            total_nodes += 1
            if isinstance(node, dict) and node.get("phase") == w["phase"]:
                correct_nodes += 1
            else:
                g_correct = False

        if g_correct:
            correct_graphs += 1

    out["graphs_labeled_correctly"] = 1.0 if correct_graphs == total_graphs else 0.0
    out["recall_at_k"] = float(correct_nodes) / float(total_nodes) if total_nodes > 0 else 0.0
    return out
