def count_partitions(graph):
    ops = graph["ops"]
    supported = set(graph["supported"])
    count = 0
    active = False
    for op in ops:
        if op in supported:
            if not active:
                count += 1
                active = True
        else:
            active = False
    return count
