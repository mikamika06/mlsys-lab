import json
import os


def generate_snapshot_file(tmp_path):
    data = {
        "active": 100,
        "allocated": 120,
        "objects": {
            "root": {"parent": None},
            "cache": {"parent": "root"}
        }
    }
    path = os.path.join(tmp_path, "snap.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path
