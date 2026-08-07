import ref


def check(workdir):
    from ep_selector.ranking import rank_execution_providers

    latencies, _, _ = ref.get_test_data()
    want_eps, want_scores = ref.rank_execution_providers(latencies)

    try:
        got_eps, got_scores = rank_execution_providers(latencies)
    except Exception as e:
        return {"ranking_matched": 0.0, "_note": f"raised {type(e).__name__}: {e}"}

    if got_eps == want_eps:
        return {"ranking_matched": 1.0}
    else:
        return {
            "ranking_matched": 0.0,
            "_note": f"expected ranking {want_eps}, got {got_eps}"
        }
