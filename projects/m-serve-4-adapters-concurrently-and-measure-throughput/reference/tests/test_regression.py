import sys
sys.path.insert(0, ".")
from loraserve.engine import run_concurrent_batch
from loraserve.metrics import compute_throughput_ratio
from loraserve.schedule import schedule_adapter_batches

def test_schedule_groups_by_adapter():
    reqs = [{"id": 1, "adapter_id": 1}, {"id": 2, "adapter_id": 2}]
    batches = schedule_adapter_batches(reqs)
    assert len(batches) == 2

def test_engine_counts_all_adapters():
    reqs = [
        {"id": 1, "adapter_id": "a1", "tokens": 10},
        {"id": 2, "adapter_id": "a2", "tokens": 10},
        {"id": 3, "adapter_id": "a3", "tokens": 10},
        {"id": 4, "adapter_id": "a4", "tokens": 10},
    ]
    res = run_concurrent_batch(reqs, ["a1", "a2", "a3", "a4"])
    assert res["active_adapters"] == 4

def test_throughput_ratio_calculation():
    multi = {"throughput": 80.0}
    base = {"throughput": 100.0}
    ratio = compute_throughput_ratio(multi, base)
    assert ratio == 0.8
