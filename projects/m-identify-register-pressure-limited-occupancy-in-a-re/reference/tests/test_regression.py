import sys

sys.path.insert(0, ".")
from profiler.occupancy import compute_occupancy
from profiler.interpreter import build_block_timing_table


def test_occupancy_bounds():
    specs = {"max_regs_per_sm": 65536, "max_threads_per_sm": 2048, "max_blocks_per_sm": 32, "warp_size": 32, "reg_allocation_unit": 256, "shmem_allocation_unit": 1024}
    res = compute_occupancy(32, 1024, 256, specs)
    assert 0.0 <= res["occupancy"] <= 1.0, f"occupancy {res['occupancy']} out of bounds"
    assert res["active_blocks"] > 0, "active blocks cannot be zero"


def test_timing_table_structure():
    table = build_block_timing_table((2, 2, 1), 5.0)
    assert len(table) == 4, f"expected 4 blocks, got {len(table)}"
    for entry in table:
        assert "block_id" in entry
        assert "duration_us" in entry
        assert entry["duration_us"] > 0.0
