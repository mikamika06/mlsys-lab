import ref

def check(workdir):
    from delegate_partition.counter import count_partitions
    graphs = ref.generate_graphs()
    ok = 0
    total = len(graphs)
    for g in graphs:
        want = ref.count_partitions(g)
        got = count_partitions(g)
        if got == want:
            ok += 1
    return {"partitions_matched": 1.0 if ok == total else 0.0}
