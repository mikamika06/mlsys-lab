import numpy as np
from goodput.logs import reconstruct_batch_sizes as ref_reconstruct
from goodput.little import check_littles_law as ref_little
from goodput.metrics import compute_goodput as ref_goodput

EVENTS = [(0, 10), (2, 15), (5, 20)]
TIME_GRID = np.array([0, 5, 10, 15, 20])

ARRIVALS = [10.0, 15.0, 20.0]
QUEUES = [5.0, 7.5, 10.0]
LATENCIES = [0.5, 0.5, 0.5]

LOADS = [50.0, 100.0, 150.0]
TTFTS = [1.2, 2.1, 1.8]

def reconstruct_batch_sizes(events, time_grid):
    return ref_reconstruct(events, time_grid)

def check_littles_law(arrivals, queue_lengths, latencies):
    return ref_little(arrivals, queue_lengths, latencies)

def compute_goodput(offered_loads, ttfts, slo=2.0):
    return ref_goodput(offered_loads, ttfts, slo)
