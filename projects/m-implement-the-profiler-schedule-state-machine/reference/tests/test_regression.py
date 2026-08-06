import sys

sys.path.insert(0, ".")
from profiler.diagnose import diagnose_zero_gpu_events
from profiler.schedule import get_step_action, schedule_summary


def test_last_active_step_is_record_and_save():
    action = get_step_action(step=3, skip_first=1, wait=1, warmup=0, active=2, repeat=1)
    assert action == "RECORD_AND_SAVE"


def test_schedule_summary_totals_match():
    summary = schedule_summary(total_steps=10, skip_first=1, wait=1, warmup=1, active=2, repeat=1)
    total = summary["none_count"] + summary["warmup_count"] + summary["record_count"] + summary["record_and_save_count"]
    assert total == 10
    assert summary["record_and_save_count"] == 1


def test_diagnose_missing_cuda():
    config = {
        "activities": ["CPU"],
        "schedule": {"skip_first": 0, "wait": 0, "warmup": 0, "active": 1, "repeat": 1},
        "total_steps": 5,
        "stepped": True,
    }
    diag = diagnose_zero_gpu_events(config)
    assert diag == "MISSING_CUDA_ACTIVITY"
