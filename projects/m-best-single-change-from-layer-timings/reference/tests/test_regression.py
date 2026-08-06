import os
import tempfile
import json
from trtexplore.analyze import best_single_change


def test_best_single_change():
    prof = {
        "layers": [
            {"name": "Conv1", "timeMs": 15.0},
            {"name": "Conv2", "timeMs": 25.0}
        ]
    }
    candidates = [
        {"Conv1": 10.0},
        {"Conv2": 18.0},
        {"Conv1": 8.0, "Conv2": 10.0}
    ]
    
    fd, path = tempfile.mkstemp()
    with open(path, 'w') as f:
        json.dump(prof, f)
    os.close(fd)

    try:
        idx = best_single_change(path, candidates)
        assert idx == 2
    finally:
        os.remove(path)
