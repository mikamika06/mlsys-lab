import sys
sys.path.insert(0, ".")
from blockinf.prune import select_layers_to_remove


def test_select_lowest_bi():
    scores = [0.5, 0.1, 0.9, 0.2]
    removed = select_layers_to_remove(scores, 2)
    assert removed == [1, 3], f"expected [1, 3], got {removed}"


def test_select_length():
    scores = [0.1, 0.2, 0.3, 0.4]
    removed = select_layers_to_remove(scores, 1)
    assert len(removed) == 1
