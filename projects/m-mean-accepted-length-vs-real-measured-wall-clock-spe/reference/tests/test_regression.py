import sys

sys.path.insert(0, ".")
from specprof.trace import parse_trace_events


def test_trace_parsing_distinct_phases():
    events = [
        {"name": "draft_eval", "cat": "draft", "dur": 100.0},
        {"name": "target_eval", "cat": "target", "dur": 200.0},
        {"name": "verify_tokens", "cat": "verify", "dur": 50.0},
        {"name": "rollback_kv", "cat": "overhead", "dur": 50.0},
    ]
    res = parse_trace_events(events)

    assert abs(res["draft"] - 0.25) < 1e-4, f"expected draft 0.25, got {res['draft']}"
    assert (
        abs(res["target"] - 0.50) < 1e-4
    ), f"expected target 0.50, got {res['target']}"
    assert (
        abs(res["verify"] - 0.125) < 1e-4
    ), f"expected verify 0.125, got {res['verify']}"
    assert (
        abs(res["overhead"] - 0.125) < 1e-4
    ), f"expected overhead 0.125, got {res['overhead']}"


def test_trace_parsing_rejects_merged_verify():
    events = [
        {"name": "draft_eval", "cat": "draft", "dur": 100.0},
        {"name": "target_eval", "cat": "target", "dur": 200.0},
        {"name": "verify_tokens", "cat": "verify", "dur": 100.0},
    ]
    res = parse_trace_events(events)

    assert res["verify"] > 0.0, "verify phase must be distinctly tracked"
    assert (
        abs(res["draft"] - res["verify"]) > 1e-4 or res["draft"] == 0.25
    ), "draft and verify must not be combined"
