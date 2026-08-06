from ortopt.levels import select_cheapest_level


def test_select_cheapest_level():
    latencies = [100.0, 92.0, 90.0, 89.5]
    setup_costs = [1.0, 5.0, 20.0, 50.0]
    chosen = select_cheapest_level(latencies, setup_costs, tolerance=0.05)
    best = min(latencies)
    assert latencies[chosen] <= best * 1.05
    assert chosen == 1
