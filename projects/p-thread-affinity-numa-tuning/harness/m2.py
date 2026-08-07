import ref


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from numa_tuning import affinity

    m = {"traffic_quantified": 0.0}
    try:
        topo = ref.get_mock_topo()
        allocs = {0: 0, 1: 1, 2: 0, 3: 1}
        traffic = affinity.analyze_inter_node_traffic(topo, allocs)
        if 0.0 <= traffic <= 1.0:
            m["traffic_quantified"] = 1.0
    except Exception:
        pass
    return m
