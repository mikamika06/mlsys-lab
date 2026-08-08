import sys

sys.path.insert(0, ".")
from ir_compare.counts import get_node_counts
from ir_compare.discrepancy import compute_folding_discrepancy


def test_node_counts_positive():
    for m in ["model_a", "model_b", "model_c"]:
        counts = get_node_counts(m)
        assert counts is not None
        assert counts["relay_nodes"] > 0
        assert counts["relax_nodes"] > 0
        assert counts["relay_nodes"] > counts["relax_nodes"]


def test_folding_discrepancy_non_negative():
    for m in ["model_a", "model_b", "model_c"]:
        disc = compute_folding_discrepancy(m)
        assert disc is not None
        assert "diff" in disc
        assert disc["diff"] >= 0.0
