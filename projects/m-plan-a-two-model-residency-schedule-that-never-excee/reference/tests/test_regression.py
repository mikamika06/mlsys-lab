import sys
sys.path.insert(0, ".")
from residency.planner import plan_residency
from residency.memory import verify_zero_copy
from residency.schedule import build_schedule

def test_schedule_never_exceeds_limit():
    model_a = {"weight_bytes": 1024 * 1024 * 1000, "kv_bytes": 1024 * 1024 * 500}
    model_b = {"weight_bytes": 1024 * 1024 * 800, "kv_bytes": 1024 * 1024 * 200}
    limit = 2000
    sched = plan_residency(model_a, model_b, limit)
    limit_bytes = limit * 1024 * 1024
    for row in sched:
        assert row["wired_bytes"] <= limit_bytes, f"Wired bytes {row['wired_bytes']} exceeds limit {limit_bytes}"

def test_zero_copy_properties():
    res = verify_zero_copy(1024)
    assert res["host_to_device_copies"] == 0
    assert res["unified_memory"] is True

def test_builder_compliance():
    sched = build_schedule(5, 4096)
    assert len(sched) == 5
    for row in sched:
        assert row["compliant"] is True
