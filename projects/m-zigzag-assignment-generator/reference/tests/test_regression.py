import sys
sys.path.insert(0, ".")
from zigzag.assignment import generate_zigzag_assignments
from zigzag.comm import compute_comm_volume
from zigzag.overlap import check_overlap_feasibility

def test_zigzag_partition_covers_all_tokens():
    num_tokens = 64
    world_size = 4
    assignments = generate_zigzag_assignments(num_tokens, world_size)
    seen = []
    for rank_tokens in assignments:
        seen.extend(rank_tokens)
    assert sorted(seen) == list(range(num_tokens))

def test_comm_volume_scaling():
    vol = compute_comm_volume(128, 4, 64, 2)
    assert vol > 0

def test_overlap_feasibility_logic():
    assert check_overlap_feasibility(10.0, 15.0, 0.5) is True
    assert check_overlap_feasibility(50.0, 10.0, 0.1) is False
