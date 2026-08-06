import ref

def check(workdir):
    from delegate_partition.optimizer import optimize_graph
    from delegate_partition.counter import count_partitions
    graphs = ref.generate_graphs()
    ok = 0
    total = len(graphs)
    for g in graphs:
        opt_got = optimize_graph(g)
        p_orig = count_partitions(g)
        p_opt = count_partitions(opt_got)
        if p_opt <= p_orig:
            ok += 1
    return {"reduction_matched": 1.0 if ok == total else 0.0}
