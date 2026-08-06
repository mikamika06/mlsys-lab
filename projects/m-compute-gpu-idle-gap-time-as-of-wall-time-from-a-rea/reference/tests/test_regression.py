from nsysprof.syncs import count_sync_points


def test_sync_counting():
    trace = [
        {"type": "kernel", "start": 0.0, "end": 1.0},
        {"type": "sync", "start": 1.0, "end": 1.2},
        {"type": "kernel", "start": 1.2, "end": 2.0}
    ]
    assert count_sync_points(trace) == 1
