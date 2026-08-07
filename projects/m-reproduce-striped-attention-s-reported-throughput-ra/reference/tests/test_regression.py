import sys
sys.path.insert(0, ".")
from striped.policy import assign_blocks
from striped.simulator import simulate_throughput


def test_striped_assignment_coverage():
    num_blocks = 16
    world_size = 4
    assignment = assign_blocks(num_blocks, world_size)
    all_assigned = [b for rank_blocks in assignment for b in rank_blocks]
    assert sorted(all_assigned) == list(range(num_blocks))
    for rank_blocks in assignment:
        assert len(rank_blocks) == num_blocks // world_size


def test_throughput_ratio_bounds():
    ratio = simulate_throughput(16, 4, 10.0, 2.0)
    assert ratio > 1.0
    assert ratio < 10.0
