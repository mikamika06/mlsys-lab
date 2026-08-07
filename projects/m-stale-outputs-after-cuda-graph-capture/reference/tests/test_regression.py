import sys

sys.path.insert(0, ".")
from ortgraph.capture import detect_stale_outputs, simulate_capture_and_run


def test_detects_stale_outputs_on_rebound_buffers():
    steps = [
        {"type": "capture", "bindings": {"out": [1, 2, 3]}},
        {
            "type": "replay",
            "step_id": 1,
            "output_bindings": {"out": 0x1000},
            "writes_output": False,
        },
    ]
    replay_inputs = {1: {"out": [9, 9, 9]}}
    res = simulate_capture_and_run(steps, replay_inputs)
    res[0]["expected"] = {"out": [9, 9, 9]}

    stale = detect_stale_outputs(res)
    assert 1 in stale, "Failed to detect stale output when graph replay did not update buffer"


def test_fresh_outputs_not_flagged():
    steps = [
        {"type": "capture", "bindings": {"out": [1, 2, 3]}},
        {
            "type": "replay",
            "step_id": 1,
            "output_bindings": {"out": 0x1000},
            "rebind_output": True,
        },
    ]
    replay_inputs = {1: {"out": [5, 5, 5]}}
    res = simulate_capture_and_run(steps, replay_inputs)
    res[0]["expected"] = {"out": [5, 5, 5]}

    stale = detect_stale_outputs(res)
    assert len(stale) == 0, f"Expected 0 stale steps, got {stale}"
