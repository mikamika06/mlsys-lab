from runneropts.options import compute_options
from runneropts.split import classify_options
from runneropts.reload import detect_reloads

def test_compute_options():
    mf = {"temperature": 0.5}
    req = {"temperature": 0.2}
    env = {}
    res = compute_options(mf, req, env)
    assert res["temperature"] == 0.2

def test_classify_options():
    traces = {"num_ctx": {"load_duration_changed": True}, "temperature": {"load_duration_changed": False}}
    lt, st = classify_options(traces)
    assert lt == ["num_ctx"]
    assert st == ["temperature"]

def test_detect_reloads():
    counters = [{"load_count": 0, "pid": 123}, {"load_count": 1, "pid": 124}]
    assert detect_reloads(counters) == [1]
