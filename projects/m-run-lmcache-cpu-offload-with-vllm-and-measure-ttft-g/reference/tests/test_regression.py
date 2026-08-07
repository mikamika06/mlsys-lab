import sys
sys.path.insert(0, ".")
from offload.runner import measure_ttft_gain
from offload.diagnose import diagnose_transfer_log
from offload.chunk import find_optimal_chunk_size


def test_runner_returns_dict():
    res = measure_ttft_gain([{"prompt_len": 1024, "cache_hit_rate": 0.8}], {"chunk_size": 256})
    assert isinstance(res, dict)
    assert "mean_ratio" in res


def test_diagnosis_output_count():
    logs = ["req1,12.5,5000", "req2,2.1,10000"]
    res = diagnose_transfer_log(logs)
    assert len(res) == len(logs)


def test_chunk_size_is_positive():
    lengths = [512, 1024, 2048]
    chunk = find_optimal_chunk_size(lengths, 1000.0)
    assert chunk > 0
    assert chunk in [64, 128, 256, 512, 1024, 2048]
