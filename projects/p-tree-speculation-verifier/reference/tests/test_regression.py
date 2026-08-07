import math
from speculation.tree import expected_length, tree_attention_mask, build_tree


def test_expected_length_correct():
    tokens = [1, 2, 3]
    parents = [-1, 0, 0]
    draft_probs = [0.875, 0.25, 0.5]
    target_probs = [
        [0.125, 0.875, 0.0, 0.0, 0.0],
        [0.125, 0.125, 0.25, 0.5, 0.0],
        [0.125, 0.125, 0.125, 0.125, 0.5],
        [0.125, 0.125, 0.125, 0.125, 0.5],
    ]
    el = expected_length(tokens, parents, draft_probs, target_probs)
    assert math.isclose(el, 2.53125), f"Expected 2.53125, got {el}"


def test_attention_mask_no_siblings():
    tokens, parents = build_tree([[1, 2], [1, 3]])
    mask = tree_attention_mask(parents)
    assert not mask[1, 2] and not mask[2, 1], "Siblings must not attend to each other"
