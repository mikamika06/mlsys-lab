import sys

sys.path.insert(0, ".")
from router.penalty import compute_penalty
from router.routing import select_best_node


def test_penalty_monotonicity():
    req = [1, 2, 3, 4, 5]
    p1 = compute_penalty([1, 2], req, 10.0)
    p2 = compute_penalty([1, 2, 3, 4], req, 10.0)
    assert p2 < p1, f"longer match should incur lower penalty: {p2} vs {p1}"


def test_penalty_zero_for_full_match():
    req = [1, 2, 3]
    p = compute_penalty([1, 2, 3], req, 10.0)
    assert p == 0.0, f"full match should have zero penalty, got {p}"


def test_select_best_node():
    nodes = {"node_a": [1, 2], "node_b": [1, 2, 3, 4]}
    req = [1, 2, 3, 4, 5]
    best = select_best_node(nodes, req, 5.0)
    assert best == "node_b", f"expected node_b, got {best}"
