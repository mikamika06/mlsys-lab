import sys
sys.path.insert(0, ".")
from goodput.logs import reconstruct_batch_sizes
from goodput.little import check_littles_law
from goodput.metrics import compute_goodput
import numpy as np

def test_batch_reconstruction_bounds():
    events = [(0, 5), (2, 8)]
    grid = np.array([0, 2, 4, 6, 8])
    res = reconstruct_batch_sizes(events, grid)
    assert len(res) == len(grid)
    assert all(r >= 0 for r in res)

def test_littles_law_consistency():
    arrivals = [10.0, 20.0]
    queues = [5.0, 10.0]
    latencies = [0.5, 0.5]
    err = check_littles_law(arrivals, queues, latencies)
    assert err < 1e-5

def test_goodput_slo_filtering():
    loads = [100.0, 200.0]
    ttfts = [1.5, 2.5]
    gp = compute_goodput(loads, ttfts, slo=2.0)
    assert gp[0] == 100.0
    assert gp[1] == 0.0
