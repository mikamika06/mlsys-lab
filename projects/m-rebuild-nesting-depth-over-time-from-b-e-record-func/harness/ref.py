def get_sample_data():
    events = [
        {"ph": "B", "name": "forward", "ts": 100, "pid": 1, "tid": 1},
        {"ph": "B", "name": "gemm", "ts": 120, "pid": 1, "tid": 1},
        {"ph": "E", "name": "gemm", "ts": 180, "pid": 1, "tid": 1},
        {"ph": "E", "name": "forward", "ts": 250, "pid": 1, "tid": 1}
    ]
    metadata = {
        "pid_names": {1: "TrainerProcess"},
        "tid_names": {1: "MainThread"}
    }
    timestamps = [50, 110, 150, 300]
    return events, metadata, timestamps
