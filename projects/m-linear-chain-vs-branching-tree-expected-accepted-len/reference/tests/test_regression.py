import sys
sys.path.insert(0, ".")

from speculative.metrics import expected_accepted_length_linear, expected_accepted_length_tree
from speculative.tree import verify_tree_sample

def test_linear_vs_tree_structure():
    parents = [-1, 0, 1, 2]
    probs = [0.8, 0.8, 0.8, 0.8]
    exp_lin = expected_accepted_length_linear(probs)
    exp_tree = expected_accepted_length_tree(parents, probs)
    assert abs(exp_lin - exp_tree) < 1e-6

def test_verify_tree_sample_requires_root_and_contiguity():
    parents = [-1, 0, 0, 1, 1]

    accepts_valid = [True, True, False, True, False]
    path = verify_tree_sample(parents, accepts_valid)
    assert path == [0, 1, 3]

    accepts_disconnected = [True, False, False, True, False]
    path_disc = verify_tree_sample(parents, accepts_disconnected)
    assert path_disc == [0]

def test_tree_branching_expectation():
    parents = [-1, 0, 0]
    probs = [0.5, 0.5, 0.5]
    exp_tree = expected_accepted_length_tree(parents, probs)
    assert abs(exp_tree - 0.75) < 1e-6
