def get_raw_data():
    return {
        "tensors": [
            {"id": 1, "size": 1024, "constant": True, "start": 0, "end": 10},
            {"id": 2, "size": 512, "constant": False, "start": 0, "end": 2},
            {"id": 3, "size": 512, "constant": False, "start": 1, "end": 3},
            {"id": 4, "size": 256, "constant": False, "start": 2, "end": 4},
            {"id": 5, "size": 128, "constant": False, "start": 4, "end": 5},
        ]
    }


def get_large_data():
    return {
        "tensors": [
            {"id": 1, "size": 400, "constant": True, "start": 0, "end": 10},
            {"id": 2, "size": 100, "constant": False, "start": 0, "end": 3},
            {"id": 3, "size": 100, "constant": False, "start": 2, "end": 5},
            {"id": 4, "size": 100, "constant": False, "start": 4, "end": 7},
            {"id": 5, "size": 100, "constant": False, "start": 6, "end": 9},
        ]
    }
