import sys
sys.path.insert(0, ".")
from treeprune.calib import reconstruct_tree_choices
import numpy as np

def test_reconstruct_length():
    fixture = np.array([[0.1, 0.4, 0.3, 0.1, 0.1],
                        [0.5, 0.1, 0.2, 0.1, 0.1]])
    res = reconstruct_tree_choices(fixture)
    assert len(res) == len(fixture)
    for row in res:
        assert len(row) > 1, "pruned tree choices must contain multiple branches per level"

def test_reconstruct_valid_indices():
    fixture = np.array([[0.2, 0.3, 0.4, 0.1, 0.0]])
    res = reconstruct_tree_choices(fixture)
    for row in res:
        for idx in row:
            assert 0 <= idx < 5
