def generate_test_data():
    p1 = {
        "pid": 0,
        "events": [
            {"name": "sync", "ts": 100, "dur": 10},
            {"name": "compute", "ts": 110, "dur": 20}
        ]
    }
    p2 = {
        "pid": 1,
        "events": [
            {"name": "sync", "ts": 150, "dur": 10},
            {"name": "compute", "ts": 160, "dur": 80}
        ]
    }
    return [p1, p2]
