import sys
sys.path.insert(0, ".")
from servebatch.boundaries import reconstruct_boundaries

def test_reconstructed_batches_contain_valid_arrivals():
    arrivals = [0.1, 0.12, 0.15, 0.3]
    batches = reconstruct_boundaries(arrivals, max_batch_size=2, timeout_s=0.1)
    flat = [item for b in batches for item in b]
    assert sorted(flat) == sorted(arrivals)

def test_max_batch_size_is_respected():
    arrivals = [0.0, 0.0, 0.0, 0.0, 0.0]
    batches = reconstruct_boundaries(arrivals, max_batch_size=2, timeout_s=1.0)
    for b in batches:
        assert len(b) <= 2
