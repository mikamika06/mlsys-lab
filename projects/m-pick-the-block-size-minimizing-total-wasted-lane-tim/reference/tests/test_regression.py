import sys
sys.path.insert(0, ".")
from lanewaste.metrics import calculate_wasted_lane_time
from lanewaste.optimizer import select_best_block_size

def test_zero_padding_when_exact_multiple():
    n = 1024
    block_size = 256
    overhead = 10.0
    val = calculate_wasted_lane_time(n, block_size, overhead)
    assert val == 4 * 10.0

def test_select_best_block_size_picks_argmin():
    candidates = [16, 32, 64, 128, 256, 512, 1024]
    n = 1025
    overhead = 5.0
    idx, min_val = select_best_block_size(n, candidates, overhead)
    all_vals = [calculate_wasted_lane_time(n, b, overhead) for b in candidates]
    assert idx == all_vals.index(min(all_vals))
    assert abs(min_val - min(all_vals)) < 1e-6
