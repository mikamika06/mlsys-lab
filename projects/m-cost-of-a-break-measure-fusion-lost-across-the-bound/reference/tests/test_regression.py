from breakcost.fusion import find_fusion_pairs
from breakcost.analyzer import find_lost_fusions
from breakcost.cost import compute_lost_cost


def test_fusion_analysis():
    nodes = [
        {"id": 0, "is_elementwise": True, "bytes": 100},
        {"id": 1, "is_elementwise": True, "bytes": 200},
        {"id": 2, "is_elementwise": False, "bytes": 400},
    ]
    pairs = find_fusion_pairs(nodes)
    assert len(pairs) == 1
    lost = find_lost_fusions(nodes, [1])
    assert len(lost) == 1
    cost = compute_lost_cost(nodes, lost)
    assert cost == 200
