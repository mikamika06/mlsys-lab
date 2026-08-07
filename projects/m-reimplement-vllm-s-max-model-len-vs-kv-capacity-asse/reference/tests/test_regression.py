import sys

sys.path.insert(0, ".")
from vllmlimits.triage import triage_startup_error


def test_triage_handles_capacity_error():
    config = {"max_model_len": 4096}
    res = triage_startup_error("ValueError: KV cache capacity insufficient for max_model_len", config)
    assert res["action"] == "reduce_max_model_len"
    assert res["recommended_max_model_len"] == 2048


def test_triage_handles_oom():
    config = {"max_model_len": 4096}
    res = triage_startup_error("CUDA Out of memory error during allocation", config)
    assert res["action"] == "increase_gpu_memory_utilization"
    assert res["recommended_gpu_memory_utilization"] == 0.95
