import ref

def check(workdir):
    from radixtree.cache import simulate_cache
    from radixtree.schedule import schedule_requests
    from radixtree.tree import TokenRadixTree
    traces = ref.get_test_traces()
    res_radix = simulate_cache(traces, "radix")
    res_flat = simulate_cache(traces, "flat")

    tree = TokenRadixTree()
    scheduled = schedule_requests(traces, tree, "lpm")

    valid = res_radix["hit_rate"] >= res_flat["hit_rate"] and len(scheduled) == len(traces)
    return {"cache_and_schedule_match": 1.0 if valid else 0.0}
