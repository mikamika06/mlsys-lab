import sys
sys.path.insert(0, ".")
from profiler_analysis.overlap import compute_overlap_percentage
from profiler_analysis.parser import parse_comm_compute_split
from profiler_analysis.classifier import classify_overlap


def test_overlap_range():
    trace = {"traceEvents": [{"cat": "compute", "ts": 0, "dur": 100}, {"cat": "comm", "ts": 10, "dur": 50}]}
    pct = compute_overlap_percentage(trace)
    assert 0.0 <= pct <= 100.0, f"overlap percentage {pct} out of bounds"


def test_parser_non_negative():
    trace = {"traceEvents": [{"cat": "compute", "ts": 0, "dur": 100}, {"cat": "comm", "ts": 100, "dur": 50}]}
    res = parse_comm_compute_split(trace)
    assert res["compute_time"] >= 0.0
    assert res["comm_time"] >= 0.0


def test_classifier_valid_output():
    trace = {"traceEvents": [{"cat": "compute", "ts": 0, "dur": 100}, {"cat": "comm", "ts": 10, "dur": 50}]}
    label = classify_overlap(trace)
    assert label in ("enabled", "disabled")
