import sys

sys.path.insert(0, ".")
from diag.labels import label_workloads
from diag.sync import find_hidden_syncs


def test_sync_detection():
    logs = [
        {"event": "kernel_launch", "ts_us": 10.0, "is_blocking": False},
        {"event": "item_call", "ts_us": 12.0, "is_blocking": True},
    ]
    syncs = find_hidden_syncs(logs)
    assert len(syncs) == 1
    assert syncs[0] == 1


def test_label_classification():
    trace = [
        {"name": "aten::add", "dur_us": 10.0, "launch_delay_us": 20.0, "flops": 100, "bytes": 200},
        {"name": "aten::empty", "dur_us": 1.0, "launch_delay_us": 1.0, "flops": 0, "bytes": 0},
    ]
    labels = label_workloads(trace)
    assert labels == ["launch_bound", "overhead"]
