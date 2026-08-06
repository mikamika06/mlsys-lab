import numpy as np
from treespec.builder import build_tree_mask_and_positions

def test_tree_mask_prevents_sibling_attention():
    parents = [-1, 0, 0]
    root_pos = 10
    mask, pos = build_tree_mask_and_positions(parents, root_pos)
    
    assert mask[1, 2] == 0
    assert mask[2, 1] == 0
    assert mask[1, 0] == 1
    assert mask[2, 0] == 1
