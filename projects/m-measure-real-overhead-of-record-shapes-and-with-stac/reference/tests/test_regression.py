import sys
sys.path.insert(0, ".")

from profoverhead.analysis import detect_missing_schedule


def test_detect_missing_schedule_catches_unscheduled():
    scheduled_events = [
        {"step": 1, "name": "forward"},
        {"step": 1, "name": "backward"},
        {"step": 2, "name": "forward"},
        {"step": 2, "name": "backward"},
    ]
    res_good = detect_missing_schedule(scheduled_events, total_steps=10, expected_active_steps=2)
    assert not res_good["missing_schedule"], "Scheduled trace should not be flagged"

    unscheduled_events = [
        {"step": i, "name": "op"} for i in range(10)
    ]
    res_bad = detect_missing_schedule(unscheduled_events, total_steps=10, expected_active_steps=2)
    assert res_bad["missing_schedule"], "Unscheduled full-run trace must be flagged"
