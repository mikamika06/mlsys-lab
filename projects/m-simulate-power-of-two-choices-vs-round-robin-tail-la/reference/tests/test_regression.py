import sys
sys.path.insert(0, ".")
import numpy as np
from routing.sim import simulate_power_of_two, simulate_round_robin
from routing.metrics import compute_tail_latency, get_per_replica_counts

def test_power_of_two_reduces_tail_latency():
    rng = np.random.RandomState(42)
    reqs = list(np.cumsum(rng.exponential(scale=0.1, size=300)))
    serv = list(rng.exponential(scale=3.0, size=300) + 0.1)
    rr_p99 = compute_tail_latency(4, reqs, serv, strategy="round_robin", percentile=99, seed=42)
    p2_p99 = compute_tail_latency(4, reqs, serv, strategy="power_of_two", percentile=99, seed=42)
    assert p2_p99 < rr_p99

def test_per_replica_counts_sum_to_total_requests():
    reqs = [0.0, 1.0, 2.0, 3.0]
    serv = [1.0, 1.0, 1.0, 1.0]
    counts = get_per_replica_counts(4, reqs, serv, strategy="power_of_two", seed=42)
    assert sum(counts) == len(reqs)
