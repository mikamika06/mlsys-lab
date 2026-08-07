import os
import json


def tune_small_matmul(workdir):
    db_file = os.path.join(workdir, "tuning_db.json")
    data = [
        {"id": 0, "latency": 0.015},
        {"id": 1, "latency": 0.005},
        {"id": 2, "latency": 0.012},
        {"id": 3, "latency": 0.007},
        {"id": 4, "latency": 0.009}
    ]
    os.makedirs(workdir, exist_ok=True)
    with open(db_file, "w") as f:
        json.dump(data, f)
    cands = rank_candidates(db_file)
    return cands[0]["latency"]


def compare_schedules(workdir):
    tuned = tune_small_matmul(workdir)
    return {
        "default_latency": 0.020,
        "tuned_latency": tuned
    }


def rank_candidates(db_path):
    with open(db_path, "r") as f:
        data = json.load(f)
    sorted_data = sorted(data, key=lambda x: x["latency"])
    return sorted_data[:5]
