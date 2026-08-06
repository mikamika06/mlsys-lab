import sys
sys.path.insert(0, ".")
from triton_prof.ncu import analyze_occupancy
from triton_prof.interpret import build_block_table

def test_occupancy_analysis():
    res = analyze_occupancy({"regs_per_thread": 64, "threads_per_block": 256, "smem_bytes": 1024, "warp_size": 32})
    assert res["bottleneck"] == "register"
    assert res["max_active_blocks"] > 0

def test_block_table_construction():
    table = build_block_table((4,), [10.5, 12.0, 11.2, 9.8])
    assert len(table) == 4
    assert table[0]["duration"] == 10.5
