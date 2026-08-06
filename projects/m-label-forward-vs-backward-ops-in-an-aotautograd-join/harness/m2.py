import ref


def check(workdir):
    from aot_tools.recompute_planner import evaluate_recompute_tradeoff

    out = {"savings_calculated_correctly": 0.0, "optimal_decisions_ratio": 0.0}
    total_graphs = len(ref.GRAPHS)
    correct_savings = 0
    correct_decisions = 0

    for g in ref.GRAPHS:
        want = ref.evaluate_recompute_tradeoff(g, max_recompute_cost=100)
        got = evaluate_recompute_tradeoff(g, max_recompute_cost=100)

        if not isinstance(got, dict):
            continue

        if got.get("saved_bytes") == want["saved_bytes"]:
            correct_savings += 1
        if got.get("decisions") == want["decisions"]:
            correct_decisions += 1

    out["savings_calculated_correctly"] = float(correct_savings) / float(total_graphs)
    out["optimal_decisions_ratio"] = float(correct_decisions) / float(total_graphs)
    return out
