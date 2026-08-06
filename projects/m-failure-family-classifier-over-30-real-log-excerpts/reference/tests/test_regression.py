import sys
sys.path.insert(0, ".")
from faildebug.timeline import reorder_logs

def test_reorder_logs_chronological():
    logs = [
        {"timestamp": 100, "rank": 1, "msg": "b"},
        {"timestamp": 50, "rank": 0, "msg": "a"},
        {"timestamp": 100, "rank": 0, "msg": "c"}
    ]
    res = reorder_logs(logs)
    assert res[0]["msg"] == "a"
    assert res[1]["msg"] == "c"
    assert res[2]["msg"] == "b"
