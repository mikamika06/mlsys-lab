import sys
sys.path.insert(0, ".")
from oom_triage.analyzer import analyze_fragmentation, find_leaked_tensors

def test_fragmentation():
    snap = {
        "segments": [{
            "size": 50,
            "blocks": [
                {"id": 1, "size": 10, "state": "allocated"},
                {"id": 0, "size": 40, "state": "free"}
            ]
        }]
    }
    res = analyze_fragmentation(snap)
    assert res["allocated"] == 10
    assert res["reserved"] == 50
    assert res["max_free"] == 40

def test_leak():
    snaps = [
        {"segments": [{"blocks": [{"id": 1, "state": "allocated"}, {"id": 2, "state": "allocated"}]}]},
        {"segments": [{"blocks": [{"id": 1, "state": "allocated"}]}]}
    ]
    assert find_leaked_tensors(snaps) == [1]
