import sys

sys.path.insert(0, ".")
from fsdp_analyzer.communication import compute_per_step_communication_bytes
from fsdp_analyzer.sharding import compute_world_shard_distribution


def test_sharding_remainder_policy():
    dist = compute_world_shard_distribution(103, 8)
    assert len(dist) == 8
    assert sum(dist) == 103
    assert dist == [13, 13, 13, 13, 13, 13, 13, 12]


def test_communication_volume_ordering():
    num_params = 1_000_000
    world_size = 8
    full = compute_per_step_communication_bytes(num_params, world_size, "FULL_SHARD")
    shard_grad = compute_per_step_communication_bytes(num_params, world_size, "SHARD_GRAD_OP")
    no_shard = compute_per_step_communication_bytes(num_params, world_size, "NO_SHARD")

    assert full > shard_grad > no_shard
    assert full == 2 * shard_grad - no_shard
