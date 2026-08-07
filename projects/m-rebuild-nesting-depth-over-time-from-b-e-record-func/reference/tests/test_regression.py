import sys
sys.path.insert(0, ".")
from profparse.parser import parse_events, map_tracks
from profparse.timeline import compute_nesting_depth

def test_b_e_pairing_accuracy():
    events = [
        {"ph": "B", "name": "outer", "ts": 10, "pid": 1, "tid": 1},
        {"ph": "B", "name": "inner", "ts": 20, "pid": 1, "tid": 1},
        {"ph": "E", "name": "inner", "ts": 40, "pid": 1, "tid": 1},
        {"ph": "E", "name": "outer", "ts": 50, "pid": 1, "tid": 1}
    ]
    res = parse_events(events)
    assert len(res) == 2
    assert res[0]["dur"] == 20
    assert res[1]["dur"] == 40

def test_nesting_depth_at_timestamps():
    events = [
        {"ph": "X", "name": "task", "ts": 10, "dur": 30, "pid": 1, "tid": 1},
        {"ph": "X", "name": "sub", "ts": 15, "dur": 10, "pid": 1, "tid": 1}
    ]
    depths = compute_nesting_depth(events, [5, 12, 20, 50])
    assert depths == [0, 1, 2, 0]

def test_track_mapping_names():
    events = [{"pid": 100, "tid": 200, "name": "foo", "ts": 0, "ph": "X", "dur": 10}]
    meta = {"pid_names": {100: "GPU 0"}, "tid_names": {200: "Stream 7"}}
    res = map_tracks(events, meta)
    assert res[0]["process_name"] == "GPU 0"
    assert res[0]["thread_name"] == "Stream 7"
