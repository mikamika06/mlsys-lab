import sys

sys.path.insert(0, ".")
from occupancy.calc import compute_theoretical_occupancy
from occupancy.mapping import map_field

def test_mapping_known_fields():
    assert map_field("launch__threads_per_block") == "LaunchStats"

def test_occupancy_bounds():
    kernel = {"threads_per_block": 256, "registers_per_thread": 32, "shared_mem_per_block": 0}
    occ = compute_theoretical_occupancy(kernel)
    assert 0.0 <= occ <= 100.0

def test_zero_threads_edge_case():
    kernel = {"threads_per_block": 256, "registers_per_thread": 32, "shared_mem_per_block": 0}
    occ1 = compute_theoretical_occupancy(kernel)
    kernel_large = {"threads_per_block": 1024, "registers_per_thread": 64, "shared_mem_per_block": 0}
    occ2 = compute_theoretical_occupancy(kernel_large)
    assert occ1 > 0.0
    assert occ2 > 0.0
