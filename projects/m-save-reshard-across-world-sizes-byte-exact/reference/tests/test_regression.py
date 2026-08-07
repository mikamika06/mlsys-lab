import sys
import numpy as np

sys.path.insert(0, ".")
from dcp.shard import compute_rank_file_size
from dcp.memory import estimate_memory_and_time
from dcp.reshard import reshard_state_dict


def test_rank_file_size_sum_matches_total():
    shapes = [(1024, 1024), (512, 512)]
    world_size = 4
    dtype_bytes = 4
    total_bytes = sum(compute_rank_file_size(shapes, world_size, r, dtype_bytes) for r in range(world_size))
    expected_total = sum(int(np.prod(s)) for s in shapes) * dtype_bytes
    assert total_bytes == expected_total, f"Sum of rank sizes {total_bytes} != expected {expected_total}"


def test_memory_estimate_scaling():
    total_bytes = 1024 * 1024 * 1024
    mem_8, _ = estimate_memory_and_time(total_bytes, 8, 10.0)
    mem_4, _ = estimate_memory_and_time(total_bytes, 4, 10.0)
    assert mem_8 == total_bytes // 8
    assert mem_4 == total_bytes // 4
    assert mem_8 < mem_4


def test_reshard_byte_exact_roundtrip():
    source_sd = [{"chunk": np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)}]
    resharded = reshard_state_dict(source_sd, 1, 2)
    reconstructed = np.concatenate([d["chunk"] for d in resharded])
    expected = source_sd[0]["chunk"]
    assert np.array_equal(reconstructed, expected), "Resharded data does not match original byte-exact"
