import sys
sys.path.insert(0, ".")
from latency.analyzer import find_losing_batch_size
from latency.tuning import tune_max_num_seqs
from latency.warmup import detect_missing_warmup

def test_losing_batch_size_basic():
    table = [
        {"batch_size": 1, "latency": 10.0, "throughput": 100.0},
        {"batch_size": 2, "latency": 15.0, "throughput": 180.0},
        {"batch_size": 4, "latency": 55.0, "throughput": 200.0},
    ]
    assert find_losing_batch_size(table, 50.0) == 4

def test_tune_max_num_seqs_basic():
    latencies = [20.0, 30.0, 45.0, 80.0]
    seqs = [16, 32, 64, 128]
    assert tune_max_num_seqs(latencies, 50.0, seqs) == 64

def test_detect_missing_warmup_basic():
    latencies = [150.0, 20.0, 21.0, 19.0]
    assert detect_missing_warmup(latencies, 2.0) is True
