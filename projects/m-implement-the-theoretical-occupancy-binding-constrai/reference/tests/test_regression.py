import sys
sys.path.insert(0, ".")
from occupancy.calc import compute_theoretical_occupancy, find_optimal_register_cap
from occupancy.ncu import cross_check_report


def test_known_config_occupancy():
    cfg = {"block_size": 256, "regs_per_thread": 32, "smem_bytes": 1024}
    occ, binding = compute_theoretical_occupancy(cfg)
    assert occ == 1.0


def test_optimal_register_cap_selection():
    cfg = {"block_size": 256, "regs_per_thread": 64, "smem_bytes": 1024}
    cap = find_optimal_register_cap(cfg, 48)
    assert cap <= 48


def test_ncu_cross_check_tolerance():
    rep = {"theoretical_occupancy": 0.75}
    assert cross_check_report(rep, 0.73) is True
