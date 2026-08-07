"""Check Milestone 1: Reproduction of RSS growth from lazy graph retention."""

import ref


def check(workdir):
    from mlxgraph.graph import LazyGraphNode, evaluate_and_clean_graph, simulate_lazy_graph_retention

    out = {"rel_err": 1.0, "rss_growth_ratio": 0.0}

    retained = simulate_lazy_graph_retention(ref.STEPS, ref.ARRAY_SIZE, retain_references=True)
    cleared = simulate_lazy_graph_retention(ref.STEPS, ref.ARRAY_SIZE, retain_references=False)

    if not retained or not cleared or len(retained) != ref.STEPS:
        out["_note"] = "simulate_lazy_graph_retention returned invalid step history"
        return out

    expected_rss = ref.expected_retained_rss(ref.STEPS, ref.ARRAY_SIZE)
    got_rss = [r["rss_bytes"] for r in retained]

    rel_errors = [abs(g - e) / float(e) for g, e in zip(got_rss, expected_rss)]
    max_err = float(max(rel_errors)) if rel_errors else 1.0
    out["rel_err"] = max_err

    first_rss = retained[0]["rss_bytes"]
    final_rss = retained[-1]["rss_bytes"]
    out["rss_growth_ratio"] = float(final_rss) / float(max(1, first_rss))

    test_nodes = []
    p = None
    for _ in range(3):
        n = LazyGraphNode((128, 128), parent=p)
        test_nodes.append(n)
        p = n

    freed = evaluate_and_clean_graph(test_nodes)
    if freed <= 0 or len(test_nodes) != 0:
        out["_note"] = "evaluate_and_clean_graph did not properly clear nodes or report freed memory"
        out["rss_growth_ratio"] = 0.0

    return out
