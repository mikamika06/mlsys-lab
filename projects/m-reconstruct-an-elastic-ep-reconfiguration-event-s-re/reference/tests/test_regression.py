import sys

sys.path.insert(0, ".")
from eplb.reconstruct import apply_event, reconstruct_layout


def test_move_preserves_total_replicas():
    layout = {0: [1, 2], 1: [3, 4]}
    event = {"type": "move", "expert": 2, "source": 0, "dest": 1}
    new_layout = apply_event(layout, event)

    old_count = sum(len(v) for v in layout.values())
    new_count = sum(len(v) for v in new_layout.values())

    assert old_count == new_count, "Replica count changed during move"
    assert 2 not in new_layout[0], "Expert was not removed from source"
    assert 2 in new_layout[1], "Expert was not added to dest"


def test_remove_drops_replica():
    layout = {0: [1, 2]}
    event = {"type": "remove", "expert": 1, "rank": 0}
    new_layout = apply_event(layout, event)
    assert 1 not in new_layout[0], "Expert remained after remove event"


def test_add_inserts_replica():
    layout = {0: [1]}
    event = {"type": "add", "expert": 2, "rank": 0}
    new_layout = apply_event(layout, event)
    assert 2 in new_layout[0], "Expert missing after add event"
    assert new_layout[0] == sorted(new_layout[0]), "Layout not sorted"
