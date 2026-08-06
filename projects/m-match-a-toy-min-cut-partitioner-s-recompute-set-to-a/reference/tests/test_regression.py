import sys
sys.path.insert(0, ".")
from partitioner.toy import compute_toy_recompute_set
import ref


def test_recompute_set_is_non_empty():
    model = ref.get_test_module()
    s = compute_toy_recompute_set(model)
    assert len(s) > 0, "recompute set cannot be empty"


def test_recompute_contains_expected_nodes():
    model = ref.get_test_module()
    s = compute_toy_recompute_set(model)
    assert "linear1" in s, "linear1 must be in recompute set"
