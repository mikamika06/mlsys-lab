import moeplan.placement as p


def test_placement_imbalance():
    loads = [100, 10, 10, 10, 100, 10, 10, 10]
    ranks = 2
    placement = p.pack_experts(loads, ranks, 100, 500)
    rank_loads = [0] * ranks
    for exp_id, rank in enumerate(placement):
        rank_loads[rank] += loads[exp_id]
    assert max(rank_loads) <= 130
