import ref


def check(workdir):
    from dist.cost import ring_cost, tree_cost, find_crossover

    alpha = 0.0001
    beta = 1e-9
    ws = 4
    sz = 1024

    rc_got = ring_cost(sz, ws, alpha, beta)
    rc_want = ref.reference_ring_cost(sz, ws, alpha, beta)

    tc_got = tree_cost(sz, ws, alpha, beta)
    tc_want = ref.reference_tree_cost(sz, ws, alpha, beta)

    co_got = find_crossover(ws, alpha, beta)
    co_want = ref.reference_find_crossover(ws, alpha, beta)

    cost_match = 1.0 if abs(rc_got - rc_want) < 1e-5 and abs(tc_got - tc_want) < 1e-5 else 0.0
    crossover_match = 1.0 if abs(co_got - co_want) < 1.0 else 0.0

    out = {
        "cost_match": float(cost_match),
        "crossover_match": float(crossover_match)
    }
    return out
